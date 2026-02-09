# %%
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.INFO)

# %%
class CNN(nn.Module):
    def __init__(self, input_dim, out_dim, kernel):
        super(CNN,self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, 
                      out_channels=out_dim, 
                      kernel_size=kernel, 
                      stride=1, padding='same', 
                      bias=True),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.02)
        )
        
    def forward(self, x):
        output = self.conv1d(x)
        return output

class TextCNN(nn.Module):
    def __init__(self, input_dim, out_dim, kernel=[]):
        super(TextCNN,self).__init__()
        layer = []
        for i,os in enumerate(kernel):
            layer.append(CNN(input_dim, out_dim, os))
        self.layer = nn.ModuleList(layer)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        o1 = self.layer[0](x).permute(0, 2, 1)
        o2 = self.layer[1](x).permute(0, 2, 1)
        o3 = self.layer[2](x).permute(0, 2, 1)
        o4 = self.layer[3](x).permute(0, 2, 1)
        return o1, o2, o3, o4

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class EnvAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_main, x_env):  # x_main: (B, Lq, D), x_env: (B, Lkv, D)
        q = self.query_proj(x_main)
        k = self.key_proj(x_env)
        v = self.value_proj(x_env)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)

def run_batched_gcn(feat, edge, mask, gcn1, gcn2):
    """
    GCN编码，逐样本循环执行，避免大图拼接造成显存爆炸

    Args:
        feat: Tensor (B, L, D)
        edge: Tensor (B, L, L) 邻接矩阵（0/1）
        mask: Tensor (B, L)
    
    Returns:
        encoded: Tensor (B, L, D)
    """
    B, L, D = feat.shape
    device = feat.device

    padded = torch.zeros_like(feat)  # 直接创建 (B, L, D) 结构

    for i in range(B):
        node_mask = mask[i] > 0  # bool tensor (L,)
        node_feat = feat[i][node_mask]  # (n_nodes, D)
        n_nodes = node_feat.size(0)
        if n_nodes == 0:
            continue

        adj = edge[i][node_mask][:, node_mask]  # 取子图邻接矩阵 (n_nodes, n_nodes)
        idx = torch.nonzero(adj > 0, as_tuple=False).T  # 边索引 shape (2, n_edges)
        if idx.size(1) == 0:
            # 空图fallback，避免报错
            idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)
            if n_nodes < 2:
                idx = torch.zeros((2,0), dtype=torch.long, device=device)  # 若节点不足2，则无边

        # ----- GCN with Residual -----
        x1 = F.relu(gcn1(node_feat, idx))  # First GCN layer
        x2 = gcn2(x1, idx)                 # Second GCN layer
        x = F.relu(node_feat + x2)         # Residual connection with input

        # 放回原位，未mask的填0
        padded[i][node_mask] = x

    return padded

def align_features(target, env, indices, mask):
    """
    对主链target与多个结构对齐env进行融合（残基级对齐+mask控制）
    
    Args:
        target: Tensor, shape (B, L, D)
        env: Tensor, shape (B, N, L, D)
        indices: LongTensor, shape (B, N, L), -1表示未对齐, -2表示补齐环境
        mask: FloatTensor, shape (B, N, L), 1保留target，0用对齐值
        
    Returns:
        Tensor (B, L, D) 融合特征
    """
    B, N, L, D = env.shape
    
    # 标记有效环境（-2表示无效，置0，否则置1）
    valid_env_mask = (indices != -2).float()  # (B, N, L)
    
    # 安全索引（将-1和-2替换为0防止gather报错）
    safe_indices = indices.clamp(min=0)  # (B, N, L)
    
    # gather 对齐特征
    aligned = torch.gather(env, 2, safe_indices.unsqueeze(-1).expand(-1, -1, -1, D))  # (B, N, L, D)
    
    # mask扩展维度
    mask_exp = mask.unsqueeze(-1)  # (B, N, L, 1)
    valid_env_mask_exp = valid_env_mask.unsqueeze(-1)  # (B, N, L, 1)
    
    # target展开
    target_expanded = target.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, L, D)
    
    # 特征混合
    mixed = mask_exp * target_expanded + (1 - mask_exp) * aligned  # (B, N, L, D)
    
    # 用 valid_env_mask 做加权求和，排除无效环境
    weighted_sum = mixed * valid_env_mask_exp  # (B, N, L, D)
    
    # 归一化因子：每个残基对应的有效环境数量（避免除0）
    valid_counts = valid_env_mask_exp.sum(dim=1).clamp(min=1)  # (B, L, 1)
    
    # fused = weighted_sum / valid_counts  # (B, L, D)
    return weighted_sum, valid_counts


# %%
class DCNPA(nn.Module):
    def __init__(self):
        super(DCNPA, self).__init__()
        self.embed_seq = nn.Embedding(25,32)
        self.embed_ss = nn.Embedding(73,32)
        self.embed_two = nn.Embedding(8,32)
        self.dense_llm_pep = nn.Linear(1024, 128)
        self.dense_llm_pro = nn.Linear(1024, 128)
        self.dense_pep = nn.Linear(3, 32)
        self.dense_pro = nn.Linear(23, 32)

        # 2025-07-31测试
        self.pep_convs = TextCNN(256, 64, [3,5,7,9])
        self.pro_convs = TextCNN(256, 64, [5,10,15,20])

        self.gcn_pep_1 = SAGEConv(256, 256)
        self.gcn_pep_2 = SAGEConv(256, 256)
        self.gcn_pro_1 = SAGEConv(256, 256)
        self.gcn_pro_2 = SAGEConv(256, 256)
        
        self.pep_proj_cat = FCNet([512, 128], act='ReLU', dropout=0.2)
        self.pro_proj_cat = FCNet([512, 128], act='ReLU', dropout=0.2)

        # 多尺度 peptide transformer
        layer_small_pep = TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=64, batch_first=True)
        layer_large_pep = TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=64 * 2, batch_first=True)
        self.pep_transformer = nn.ModuleList([
            TransformerEncoder(layer_small_pep, num_layers=1),
            TransformerEncoder(layer_large_pep, num_layers=2)
        ])

        # 多尺度 protein transformer
        layer_small_pro = TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=64, batch_first=True)
        layer_large_pro = TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=64 * 2, batch_first=True)
        self.pro_transformer = nn.ModuleList([
            TransformerEncoder(layer_small_pro, num_layers=1),
            TransformerEncoder(layer_large_pro, num_layers=2)
        ])
        
        self.pep_cross_attn = EnvAttention(d_model=256)
        self.pro_cross_attn = EnvAttention(d_model=256)
        
        # 初始化为可学习参数（logits）
        self.raw_weight = nn.Parameter(torch.tensor([8.0, 1.0, 1.0, 2.0, 2.0]))  # 主交互初始更高

        # 模型 init 中添加
        self.norm_origin = nn.LayerNorm([800, 50])  # 替换成你的实际 L_pro, L_pep
        self.norm_sim_pep = nn.LayerNorm([800, 50])
        self.norm_sim_pro = nn.LayerNorm([800, 50])
        self.norm_mer_pep = nn.LayerNorm([800, 50])
        self.norm_mer_pro = nn.LayerNorm([800, 50])

        self.interaction_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, peptide_seq_feature, protein_seq_feature, peptide_ss_feature, protein_ss_feature,
                peptide_2_feature, protein_2_feature, peptide_pretrain_feature, protein_pretrain_feature, 
                peptide_dense_feature, protein_dense_feature, peptide_edge_feature, protein_edge_feature, 
                peptide_mask, protein_mask,
                peptide_simenv_seq_padded, protein_simenv_seq_padded, peptide_simenv_ss_padded, protein_simenv_ss_padded,
                peptide_simenv_2_padded, protein_simenv_2_padded, peptide_simenv_pretrain_padded, protein_simenv_pretrain_padded, 
                peptide_simenv_dense_padded, protein_simenv_dense_padded, peptide_simenv_edge_padded, protein_simenv_edge_padded,
                peptide_sim_mask, protein_sim_mask, peptide_sim_index, protein_sim_index,
                peptide_merenv_seq_padded, protein_merenv_seq_padded, peptide_merenv_ss_padded, protein_merenv_ss_padded,
                peptide_merenv_2_padded, protein_merenv_2_padded, peptide_merenv_pretrain_padded, protein_merenv_pretrain_padded,
                peptide_merenv_dense_padded, protein_merenv_dense_padded, peptide_merenv_edge_padded, protein_merenv_edge_padded):

        # ============ 主图特征嵌入 ============
        x_seq_pep = self.embed_seq(peptide_seq_feature)
        x_seq_pro = self.embed_seq(protein_seq_feature)
        x_ss_pep = self.embed_ss(peptide_ss_feature)
        x_ss_pro = self.embed_ss(protein_ss_feature)
        x_2_pep = self.embed_two(peptide_2_feature)
        x_2_pro = self.embed_two(protein_2_feature)
    
        x_llm_pep = self.dense_llm_pep(peptide_pretrain_feature)
        x_llm_pro = self.dense_llm_pro(protein_pretrain_feature)
        x_dens_pep = self.dense_pep(peptide_dense_feature)
        x_dens_pro = self.dense_pro(protein_dense_feature)

        encode_peptide = torch.cat([x_seq_pep, x_ss_pep, x_2_pep, x_llm_pep, x_dens_pep],dim=-1) * peptide_mask.unsqueeze(-1)
        encode_protein = torch.cat([x_seq_pro, x_ss_pro, x_2_pro, x_llm_pro, x_dens_pro],dim=-1) * protein_mask.unsqueeze(-1)

        # ================       TextCNN       ================
        c1_pep, c2_pep, c3_pep, c4_pep = self.pep_convs(encode_peptide)
        encode_peptide_local = torch.cat([c1_pep, c2_pep, c3_pep, c4_pep], dim=-1)
        c1_pro, c2_pro, c3_pro, c4_pro = self.pro_convs(encode_protein)
        encode_protein_local = torch.cat([c1_pro, c2_pro, c3_pro, c4_pro], dim=-1)

        # ============ GCN 编码主图 ============
        encode_peptide_space = run_batched_gcn(encode_peptide, peptide_edge_feature, peptide_mask, self.gcn_pep_1, self.gcn_pep_2)
        encode_protein_space = run_batched_gcn(encode_protein, protein_edge_feature, protein_mask, self.gcn_pro_1, self.gcn_pro_2)

        encode_peptide = torch.cat([encode_peptide_local, encode_peptide_space], dim=-1)
        encode_protein = torch.cat([encode_protein_local, encode_protein_space], dim=-1)

        encode_peptide = self.pep_proj_cat(encode_peptide) # torch.Size([8, 50, 128])
        encode_protein = self.pro_proj_cat(encode_protein) # torch.Size([8, 800, 128])

        # ================     Transformer     ================
        pep_features = [encoder(encode_peptide) for encoder in self.pep_transformer]
        pro_features = [encoder(encode_protein) for encoder in self.pro_transformer]
        # 融合所有尺度
        encode_peptide = torch.cat(pep_features, dim=-1)  # torch.Size([8, 50, 256])
        encode_protein = torch.cat(pro_features, dim=-1)  # torch.Size([8, 800, 256])

        logging.info('encode_protein: ', encode_protein)
        logging.info('encode_peptide: ', encode_peptide)

        matmul_pair_origin = torch.matmul(encode_protein, encode_peptide.transpose(1, 2))

        logging.info('matmul_pair_origin: ', matmul_pair_origin)


        # ============ 多肽相似环境图 ============
        if peptide_simenv_pretrain_padded.numel() != 0:
            B, max_m, L_pep, _ = peptide_simenv_pretrain_padded.shape
            flat_seq = self.embed_seq(peptide_simenv_seq_padded.view(B * max_m, L_pep))
            flat_ss = self.embed_ss(peptide_simenv_ss_padded.view(B * max_m, L_pep))
            flat_2 = self.embed_two(peptide_simenv_2_padded.view(B * max_m, L_pep))
            flat_llm = self.dense_llm_pep(peptide_simenv_pretrain_padded.view(B * max_m, L_pep, -1))
            flat_dens = self.dense_pep(peptide_simenv_dense_padded.view(B * max_m, L_pep, -1))

            feat = (torch.cat([flat_seq, flat_ss, flat_2, flat_llm, flat_dens], dim=-1) *
                    (peptide_simenv_pretrain_padded.max(dim=-1).values != 0).float().view(B * max_m, L_pep, 1))
            edge = peptide_simenv_edge_padded.view(B * max_m, L_pep, L_pep)

            c1_pep, c2_pep, c3_pep, c4_pep = self.pep_convs(feat)
            sim_pep_out_local = torch.cat([c1_pep, c2_pep, c3_pep, c4_pep], dim=-1)
            sim_pep_out_space = run_batched_gcn(feat, edge, feat.abs().sum(dim=-1) > 0, self.gcn_pep_1, self.gcn_pep_2)
            sim_pep_out = torch.cat([sim_pep_out_local, sim_pep_out_space], -1)
            sim_pep_out = self.pep_proj_cat(sim_pep_out)
            pep_features = [encoder(sim_pep_out) for encoder in self.pep_transformer]
            sim_pep_out = torch.cat(pep_features, dim=-1)
            sim_pep_out = sim_pep_out.view(B, max_m, L_pep, -1)

            final_feature_padded, valid_counts = align_features(encode_peptide, sim_pep_out, peptide_sim_index, peptide_sim_mask)
            matmul_pair_sim_pep = torch.matmul(encode_protein.unsqueeze(1), final_feature_padded.transpose(2, 3)).sum(dim=1)
            matmul_pair_sim_pep = (matmul_pair_sim_pep.transpose(1, 2) / valid_counts).transpose(1, 2)

        else:
            matmul_pair_sim_pep = torch.zeros_like(matmul_pair_origin)


        # ============ 蛋白相似环境图 ============
        if protein_simenv_pretrain_padded.numel() != 0:
            B, max_n, L_pro, _ = protein_simenv_pretrain_padded.shape

            flat_seq = self.embed_seq(protein_simenv_seq_padded.view(B * max_n, L_pro))
            flat_ss = self.embed_ss(protein_simenv_ss_padded.view(B * max_n, L_pro))
            flat_2 = self.embed_two(protein_simenv_2_padded.view(B * max_n, L_pro))
            flat_llm = self.dense_llm_pro(protein_simenv_pretrain_padded.view(B * max_n, L_pro, -1))
            flat_dens = self.dense_pro(protein_simenv_dense_padded.view(B * max_n, L_pro, -1))
            feat = (torch.cat([flat_seq, flat_ss, flat_2, flat_llm, flat_dens], dim=-1) *
                    (protein_simenv_pretrain_padded.max(dim=-1).values != 0).float().view(B * max_n, L_pro, 1))
            edge = protein_simenv_edge_padded.view(B * max_n, L_pro, L_pro)

            c1_pro, c2_pro, c3_pro, c4_pro = self.pro_convs(feat)
            sim_pro_out_local = torch.cat([c1_pro, c2_pro, c3_pro, c4_pro], dim=-1)
            sim_pro_out_space = run_batched_gcn(feat, edge, feat.abs().sum(dim=-1) > 0, self.gcn_pro_1, self.gcn_pro_2)
            sim_pro_out = torch.cat([sim_pro_out_local, sim_pro_out_space], -1)
            sim_pro_out = self.pro_proj_cat(sim_pro_out)
            pro_features = [encoder(sim_pro_out) for encoder in self.pro_transformer]
            sim_pro_out = torch.cat(pro_features, dim=-1)
            sim_pro_out = sim_pro_out.view(B, max_n, L_pro, -1)

            final_feature_padded, valid_counts = align_features(encode_protein, sim_pro_out, protein_sim_index, protein_sim_mask)
            matmul_pair_sim_pro = torch.matmul(final_feature_padded, encode_peptide.transpose(1, 2).unsqueeze(1)).sum(dim=1)
            matmul_pair_sim_pro = matmul_pair_sim_pro / valid_counts
        else:
            matmul_pair_sim_pro = torch.zeros_like(matmul_pair_origin)



        # ================     Environment     ================
        if peptide_merenv_pretrain_padded.numel() != 0:
            B, max_n, L_pro, D = peptide_merenv_pretrain_padded.shape
            flat_seq = self.embed_seq(peptide_merenv_seq_padded.view(B * max_n, L_pro))
            flat_ss = self.embed_ss(peptide_merenv_ss_padded.view(B * max_n, L_pro))
            flat_2 = self.embed_two(peptide_merenv_2_padded.view(B * max_n, L_pro))
            flat_llm = self.dense_llm_pro(peptide_merenv_pretrain_padded.view(B * max_n, L_pro, -1))
            flat_dens = self.dense_pro(peptide_merenv_dense_padded.view(B * max_n, L_pro, -1))
            feat = (torch.cat([flat_seq, flat_ss, flat_2, flat_llm, flat_dens], dim=-1) *
                    (peptide_merenv_pretrain_padded.max(dim=-1).values != 0).float().view(B * max_n, L_pro, 1))
            edge = peptide_merenv_edge_padded.view(B * max_n, L_pro, L_pro)

            c1_pro, c2_pro, c3_pro, c4_pro = self.pro_convs(feat)
            mer_pep_out_local = torch.cat([c1_pro, c2_pro, c3_pro, c4_pro], dim=-1)
            mer_pep_out_space = run_batched_gcn(feat, edge, feat.abs().sum(dim=-1) > 0, self.gcn_pro_1, self.gcn_pro_2)
            mer_pep_out = torch.cat([mer_pep_out_local, mer_pep_out_space], -1)
            mer_pep_out = self.pro_proj_cat(mer_pep_out)
            pro_features = [encoder(mer_pep_out) for encoder in self.pro_transformer]
            mer_pep_out = torch.cat(pro_features, dim=-1)
            mer_pep_out = mer_pep_out.view(B, max_n * L_pro, -1)
            
            # 拼接 peptide 和 pep_env_flat： [B, 50+3×800, D]
            peptide_plus_env = torch.cat([encode_peptide, mer_pep_out], dim=1)
            # protein 是 query，peptide+env 是 key/value
            fused_out = self.pep_cross_attn(encode_protein, peptide_plus_env)  # (B, 800, 128)
            # 最终输出 (B, 800, 50)
            matmul_pair_mer_pep = torch.matmul(fused_out, encode_peptide.transpose(1, 2))  # (B, 800, 50)
        else:
            matmul_pair_mer_pep = torch.zeros_like(matmul_pair_origin)
        
        if protein_merenv_pretrain_padded.numel() != 0:
            B, max_m, L_pep, _ = protein_merenv_pretrain_padded.shape
            flat_seq = self.embed_seq(protein_merenv_seq_padded.view(B * max_m, L_pep))
            flat_ss = self.embed_ss(protein_merenv_ss_padded.view(B * max_m, L_pep))
            flat_2 = self.embed_two(protein_merenv_2_padded.view(B * max_m, L_pep))
            flat_llm = self.dense_llm_pep(protein_merenv_pretrain_padded.view(B * max_m, L_pep, -1))
            flat_dens = self.dense_pep(protein_merenv_dense_padded.view(B * max_m, L_pep, -1))

            feat = (torch.cat([flat_seq, flat_ss, flat_2, flat_llm, flat_dens], dim=-1) *
                    (protein_merenv_pretrain_padded.max(dim=-1).values != 0).float().view(B * max_m, L_pep, 1))
            edge = protein_merenv_edge_padded.view(B * max_m, L_pep, L_pep)

            c1_pep, c2_pep, c3_pep, c4_pep = self.pep_convs(feat)
            mer_pro_out_local = torch.cat([c1_pep, c2_pep, c3_pep, c4_pep], dim=-1)
            mer_pro_out_space = run_batched_gcn(feat, edge, feat.abs().sum(dim=-1) > 0, self.gcn_pep_1, self.gcn_pep_2)
            mer_pro_out = torch.cat([mer_pro_out_local, mer_pro_out_space], -1)
            mer_pro_out = self.pep_proj_cat(mer_pro_out)
            pep_features = [encoder(mer_pro_out) for encoder in self.pep_transformer]
            mer_pro_out = torch.cat(pep_features, dim=-1)
            mer_pro_out = mer_pro_out.view(B, max_m * L_pep, -1)
            
            protein_plus_env = torch.cat([encode_protein, mer_pro_out], dim=1)
            fused_out = self.pro_cross_attn(encode_peptide, protein_plus_env)  # (B, 50, 128)
            # 最终输出 (B, 800, 50)
            matmul_pair_mer_pro = torch.matmul(fused_out, encode_protein.transpose(1, 2)).transpose(1, 2)  # (B, 800, 50)
        else:
            matmul_pair_mer_pro = torch.zeros_like(matmul_pair_origin)
        
        
        # ============ 交互图融合与卷积 ============        
        # forward 中调用（仅作用于非 mask 区域）
        matmul_pair_origin = self.norm_origin(matmul_pair_origin)
        matmul_pair_mer_pep = self.norm_mer_pep(matmul_pair_mer_pep)
        matmul_pair_mer_pro = self.norm_mer_pro(matmul_pair_mer_pro)

        interaction_mask = torch.matmul(protein_mask.unsqueeze(-1), peptide_mask.unsqueeze(1))
        weights = F.softmax(self.raw_weight, dim=0)
        alpha, beta1, beta2, gamma1, gamma2 = weights

        matmul_pair_combined = (
            alpha * matmul_pair_origin +
            beta1 * matmul_pair_sim_pep +
            beta2 * matmul_pair_sim_pro +
            gamma1 * matmul_pair_mer_pep +
            gamma2 * matmul_pair_mer_pro
        ) * interaction_mask

        out = self.interaction_conv(matmul_pair_combined.unsqueeze(1)).squeeze(1)
        return out * interaction_mask

# %%
