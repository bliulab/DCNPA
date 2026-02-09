# -*- coding: utf-8 -*-
import os
import sys
import csv
import torch
import pickle
import argparse
import numpy as np
from itertools import chain

from conf import *
from deepblastanalysis import DeepAnalysis
from FeatureExtract import *
from model import DCNPA
from typing import List, Dict

import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cpu")

def save_features_to_pkl(peptide_features, protein_features, save_dir):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # --- 1. 整理多肽特征 ---
    peptide_data = {
        "seq_feature": peptide_features[0],
        "2_feature": peptide_features[1],
        "dense_feature": peptide_features[2],
        "ss_feature": peptide_features[3],
        "pretrain_feature": peptide_features[4],
        "edge_feature": peptide_features[5]
    }
    
    # --- 2. 整理蛋白质特征 ---
    protein_data = {
        "seq_feature": protein_features[0],
        "2_feature": protein_features[1],
        "dense_feature": protein_features[2],
        "ss_feature": protein_features[3],
        "pretrain_feature": protein_features[4],
        "edge_feature": protein_features[5]
    }

    # --- 3. 写入文件 ---
    pep_file = os.path.join(save_dir, "peptide_features_combined.pkl")
    pro_file = os.path.join(save_dir, "protein_features_combined.pkl")

    with open(pep_file, 'wb') as f:
        pickle.dump(peptide_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(pro_file, 'wb') as f:
        pickle.dump(protein_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 多肽特征已保存至: {pep_file}")
    print(f"✅ 蛋白质特征已保存至: {pro_file}")


def run_db_in_venv(seq_list, uip, types):
    # 1. 定义专属环境的 Python 路径
    venv_python = "/data/webserver/anaconda3/envs/DeepBLAST/bin/python"
    # 2. 定义脚本路径
    script_path = "/data/www/DCNPA/webserver/deepblastsim.py"
    
    # 3. 将输入的列表存为临时 pkl 文件，供子进程读取
    tmp_pkl = os.path.join(uip, f"input_seqs_{types}.pkl")
    with open(tmp_pkl, 'wb') as f:
        pickle.dump(seq_list, f)
        
    logging.info(f"正在专属环境 {venv_python} 中启动比对任务: {types}")
    
    # 4. 构建并执行命令
    cmd = [venv_python, script_path, tmp_pkl, uip, types]
    
    try:
        # shell=False 更安全；check=True 报错时会抛出异常
        subprocess.run(cmd, check=True)
        logging.info(f"DeepBLAST {types} 任务成功执行完毕")
    except subprocess.CalledProcessError as e:
        logging.error(f"DeepBLAST 执行出错，返回码: {e.returncode}")
        raise e
    finally:
        # 清理临时 pkl 文件
        if os.path.exists(tmp_pkl):
            os.remove(tmp_pkl)


def clean_uip_directory(uip_path):
    # 确保路径存在
    if not os.path.exists(uip_path):
        print(f"路径 {uip_path} 不存在")
        return

    # 定义精确保留的文件名白名单
    keep_filenames = {'do_shap.txt', 'Peptide_Main.fasta', 'Protein_Main.fasta', 'Env_All.fasta'}

    for filename in os.listdir(uip_path):
        file_path = os.path.join(uip_path, filename)
        
        # 只处理文件，跳过文件夹
        if os.path.isfile(file_path):
            # 如果当前文件名不在白名单中，则执行删除
            if filename not in keep_filenames:
                try:
                    os.remove(file_path)
                    print(f"已删除冗余文件: {filename}")
                except Exception as e:
                    print(f"无法删除 {filename}: {e}")

def SeqMask(seqlen_list, seqlen_lim, device):
    seq_mask = torch.zeros((len(seqlen_list), seqlen_lim), dtype=torch.float32)
    for i, length in enumerate(seqlen_list):
        seq_mask[i, :length] = 1.0
    seq_mask = seq_mask.to(device)
    return seq_mask

def pad_and_stack_nested_tensor(
    seqs_nested: List[List[str]],
    feature_dict: Dict[str, np.ndarray],
    max_len: int
) -> torch.Tensor:
    """
    将嵌套序列映射到特征，自动适配一维或二维输入。
    - 若每个序列特征为 [L]，返回 [B, N, L]
    - 若每个序列特征为 [L, D]，返回 [B, N, L, D]
    """

    # 1. 处理顶级输入为空的情况
    if not seqs_nested:
        return torch.tensor([])
    
    batch_size = len(seqs_nested)
    max_num_subseq = max(len(sublist) for sublist in seqs_nested)

    # ---------- 核心：安全推断 feature 维度 ----------
    if len(feature_dict) == 0:
        # 无法推断 D，只能退化为 [B, N, L]
        result = np.zeros(
            (batch_size, max_num_subseq, max_len),
            dtype=np.float32
        )
        return torch.from_numpy(result)

    sample_feat = next(iter(feature_dict.values()))
    if sample_feat.ndim == 1:
        # 处理一维特征：每个序列是 [L]
        result = np.zeros((batch_size, max_num_subseq, max_len), dtype=np.float32)

        for i, group in enumerate(seqs_nested):
            for j, seq in enumerate(group):
                feat = feature_dict[seq]  # [L]
                length = min(len(feat), max_len)
                result[i, j, :length] = feat[:length]

        return torch.from_numpy(result)  # [B, N, L]
    elif sample_feat.ndim == 2:
        # 处理二维特征：每个序列是 [L, D]
        feature_dim = sample_feat.shape[1]
        result = np.zeros((batch_size, max_num_subseq, max_len, feature_dim), dtype=np.float32)

        for i, group in enumerate(seqs_nested):
            for j, seq in enumerate(group):
                feat = feature_dict[seq]  # [L, D]
                length = min(feat.shape[0], max_len)
                result[i, j, :length] = feat[:length]

        return torch.from_numpy(result)  # [B, N, L, D]

    else:
        raise ValueError(f"Unsupported feature shape: {sample_feat.shape}")

def pad_and_stack_contact_maps(
    seqs_nested: List[List[str]],
    contact_map_dict: Dict[str, np.ndarray],
    max_len: int
) -> torch.Tensor:
    """
    将嵌套序列的 contact map 映射并填充，返回四维张量：
    [batch, max_num_subseq, max_len, max_len]
    """

    # 1. 处理顶级输入为空的情况
    if not seqs_nested:
        return torch.tensor([])
    
    batch_size = len(seqs_nested)
    max_num_subseq = max(len(sublist) for sublist in seqs_nested)

    # 初始化结果张量
    result = np.zeros((batch_size, max_num_subseq, max_len, max_len), dtype=np.float32)

    for i, group in enumerate(seqs_nested):
        for j, seq in enumerate(group):
            if seq in contact_map_dict:
                contact = contact_map_dict[seq]
                L = min(contact.shape[0], max_len)
                result[i, j, :L, :L] = contact[:L, :L]
            else:
                # 如果没有contact map，就补主对角线
                L = min(len(seq), max_len)
                result[i, j, range(L), range(L)] = 1.0

    return torch.from_numpy(result)

def DCNPA_Predict(peptide_seq_feature, protein_seq_feature, peptide_ss_feature, protein_ss_feature,
                                                                peptide_2_feature, protein_2_feature, peptide_pretrain_feature, protein_pretrain_feature, 
                                                                peptide_dense_feature, protein_dense_feature, peptide_edge_feature, protein_edge_feature, 
                                                                peptide_mask, protein_mask,
                                                                peptide_simenv_seq_padded, protein_simenv_seq_padded, peptide_simenv_ss_padded, protein_simenv_ss_padded,
                                                                peptide_simenv_2_padded, protein_simenv_2_padded, peptide_simenv_pretrain_padded, protein_simenv_pretrain_padded, 
                                                                peptide_simenv_dense_padded, protein_simenv_dense_padded, peptide_simenv_edge_padded, protein_simenv_edge_padded,
                                                                peptide_sim_mask, protein_sim_mask, peptide_sim_index, protein_sim_index,
                                                                peptide_merenv_seq_padded, protein_merenv_seq_padded, peptide_merenv_ss_padded, protein_merenv_ss_padded,
                                                                peptide_merenv_2_padded, protein_merenv_2_padded, peptide_merenv_pretrain_padded, protein_merenv_pretrain_padded,
                                                                peptide_merenv_dense_padded, protein_merenv_dense_padded, peptide_merenv_edge_padded, protein_merenv_edge_padded,
                                                                peplen, prolen):
    

    logging.info("加载模型")
    model = DCNPA().to(device)
    model_path = os.path.join(MODEL_FOLD, 'checkpoint.pth')
    
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # --- 核心修复代码开始 ---
    state_dict = checkpoint["model_state_dict"]
    
    # 创建一个新的 dict，去掉 'module.' 前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 去掉 'module.' (长度为7)
        new_state_dict[name] = v
    # --- 核心修复代码结束 ---

    # 使用处理后的 state_dict 加载
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()



    logging.info("开始预测")
    ouputs_all = model(peptide_seq_feature, protein_seq_feature, peptide_ss_feature, protein_ss_feature,
                    peptide_2_feature, protein_2_feature, peptide_pretrain_feature, protein_pretrain_feature, 
                    peptide_dense_feature, protein_dense_feature, peptide_edge_feature, protein_edge_feature, 
                    peptide_mask, protein_mask,
                    peptide_simenv_seq_padded, protein_simenv_seq_padded, peptide_simenv_ss_padded, protein_simenv_ss_padded,
                    peptide_simenv_2_padded, protein_simenv_2_padded, peptide_simenv_pretrain_padded, protein_simenv_pretrain_padded, 
                    peptide_simenv_dense_padded, protein_simenv_dense_padded, peptide_simenv_edge_padded, protein_simenv_edge_padded,
                    peptide_sim_mask, protein_sim_mask, peptide_sim_index, protein_sim_index,
                    peptide_merenv_seq_padded, protein_merenv_seq_padded, peptide_merenv_ss_padded, protein_merenv_ss_padded,
                    peptide_merenv_2_padded, protein_merenv_2_padded, peptide_merenv_pretrain_padded, protein_merenv_pretrain_padded,
                    peptide_merenv_dense_padded, protein_merenv_dense_padded, peptide_merenv_edge_padded, protein_merenv_edge_padded)
    
    logging.info("提取多肽及蛋白质预测结合残基")
    ouputs_all = ouputs_all.detach().numpy()
    ouputs_all = ouputs_all[0, :prolen[0], :peplen[0]]  # (prolen, peplen)
    # print(ouputs_all.shape)  # (15, 133)
    
    # 不区分非共价键类型的多肽及蛋白质结合残基预测结果
    ouputs_all_pep = np.max(ouputs_all, 0)  # (peplen,)
    ouputs_all_pro = np.max(ouputs_all, 1)  # (prolen,)
    # print(ouputs_all_pep.shape)  # (15,)
    # print(ouputs_all_pro.shape)  # (133,)
    
    return ouputs_all, ouputs_all_pep, ouputs_all_pro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 2025-09-27 新增shap分析
    parser.add_argument("-uip", required=True, help="User dir")
    parser.add_argument("-do_shap", default="0", help="Whether to run SHAP (0/1)")

    args = parser.parse_args()
    uip = args.uip
    
    # 定义文件路径
    pep_uip = os.path.join(uip, 'Peptide_Main.fasta')
    pro_uip = os.path.join(uip, 'Protein_Main.fasta')
    env_uip = os.path.join(uip, 'Env_All.fasta') # 新增：环境分子文件
    
    # --- 读取主序列 ---
    pep_seq_list, pro_seq_list = [], []
    for i in open(pep_uip):
        if i[0] != '>':
            pep_seq_list.append(i.strip().upper())
    for i in open(pro_uip):
        if i[0] != '>':
            pro_seq_list.append(i.strip().upper())

    # 并行提取蛋白质和多肽的特征
    (protein_features, peptide_features) = parallel_feature_extract(pro_seq_list, pep_seq_list, uip)
    
    # save_features_to_pkl(peptide_features, protein_features, uip)

    clean_uip_directory(uip) # 清理特征的中间文件

    # 拆分结果
    protein_seq_feature, protein_2_feature, protein_dense_feature, protein_ss_feature, protein_pretrain_feature, protein_edge_feature = protein_features
    peptide_seq_feature, peptide_2_feature, peptide_dense_feature, peptide_ss_feature, peptide_pretrain_feature, peptide_edge_feature = peptide_features

    

    pep_len = [len(pep_seq_list[0])]
    pro_len = [len(pro_seq_list[0])]

    peptide_mask = SeqMask(pep_len, 50, device)  # torch.Size([16, 50])
    protein_mask = SeqMask(pro_len, 800, device)  # torch.Size([16, 800])
    


    # =========================== 2 目标自适应动态语境网络 ===========================
    # 用勾选的结果来判断是否执行这一部分，默认走不执行就可以了
    do_shap = args.do_shap.lower() in ["1", "true", "yes"]
    logging.info(f"do_shap={do_shap}")
    

    if do_shap:
        logging.info("目标自适应动态语境网络执行中")
        logging.info("多肽相似分子查询")
        run_db_in_venv(pep_seq_list, uip, 'peptide')
        logging.info("蛋白质相似分子查询")
        run_db_in_venv(pro_seq_list, uip, 'protein')

        # 加载其他已经提取好的比对数据库的特征
        base_path = '/data/www/DCNPA/webserver/savefeatures/subset_test/'
        prefixes = ['peptide', 'protein']
        # 修改为更清晰的映射关系
        feat_mapping = {
            '': 'feature', 
            'ss_': 'ss_feature', 
            '2_': '2_feature', 
            'T5_': 'T5_feature', 
            'dense_': 'dense_feature', 
            'edge_': 'edge_feature'
        }

        # 1. 统一加载到一个大字典里
        feats = {}
        for pref in prefixes:
            for ft, label in feat_mapping.items():
                file_name = f"{pref}_{label}_dict.pkl"
                file_path = os.path.join(base_path, file_name)
                
                # 存入字典，例如 key 为 'peptide_feature'
                dict_key = f"{pref}_{label}"
                with open(file_path, 'rb') as f:
                    feats[dict_key] = pickle.load(f, encoding='iso-8859-1')

        logging.info("相似分子特征加载完毕，可供调用。")

        # 加载pkl文件，生成相似的序列 mask 和 index
        peptide_simenv_seq, peptide_sim_mask, peptide_sim_index = DeepAnalysis(pep_seq_list, os.path.join(uip, 'peptide_similarity_dict.pkl'), feats['peptide_T5_feature'])
        protein_simenv_seq, protein_sim_mask, protein_sim_index = DeepAnalysis(pro_seq_list, os.path.join(uip, 'protein_similarity_dict.pkl'), feats['protein_T5_feature'])

        # 提取比对相似分子对应的特征
        peptide_simenv_seq_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_feature'], 50).long()
        protein_simenv_seq_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_feature'], 800).long()
        peptide_simenv_ss_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_ss_feature'], 50).long()
        protein_simenv_ss_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_ss_feature'], 800).long()
        peptide_simenv_2_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_2_feature'], 50).long()
        protein_simenv_2_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_2_feature'], 800).long()
        peptide_simenv_pretrain_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_T5_feature'], 50)
        protein_simenv_pretrain_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_T5_feature'], 800)
        peptide_simenv_dense_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_dense_feature'], 50)
        protein_simenv_dense_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_dense_feature'], 800)
        peptide_simenv_edge_padded = pad_and_stack_contact_maps(peptide_simenv_seq, feats['peptide_edge_feature'], 50)
        protein_simenv_edge_padded = pad_and_stack_contact_maps(protein_simenv_seq, feats['protein_edge_feature'], 800)

        logging.info("目标自适应动态语境网络处理完毕")

    else:
        logging.info("不执行目标自适应动态语境网络")

        peptide_simenv_seq, protein_simenv_seq = [[]], [[]]

        peptide_simenv_seq_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50).long()
        protein_simenv_seq_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, 800).long()
        peptide_simenv_ss_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50).long()
        protein_simenv_ss_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, 800).long()
        peptide_simenv_2_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50).long()
        protein_simenv_2_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, 800).long()
        peptide_simenv_pretrain_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50)
        protein_simenv_pretrain_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, 800)
        peptide_simenv_dense_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50)
        protein_simenv_dense_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, 800)
        peptide_simenv_edge_padded = pad_and_stack_contact_maps(peptide_simenv_seq, {}, 50)
        protein_simenv_edge_padded = pad_and_stack_contact_maps(protein_simenv_seq, {}, 800)

        peptide_sim_mask, peptide_sim_index, protein_sim_mask, protein_sim_index = [], [], [], []

    # sys.exit(0)


    # =========================== 3 多聚体感知动态语境网络 ===========================
    # --- 读取环境分子并分流 ---
    peptide_merenv_seq = [[]] # 保持你之前的嵌套格式需求
    protein_merenv_seq = [[]]
    
    if os.path.exists(env_uip):
        with open(env_uip, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('>'):
                    continue
                
                seq = line.upper()
                # 按照 50 的阈值进行分流
                if len(seq) <= 50:
                    protein_merenv_seq[0].append(seq)
                else:
                    peptide_merenv_seq[0].append(seq)

    # 3-2 准备存储多聚体环境序列特征的字典（如果还没初始化）
    protein_feature_dict_merenv, protein_ss_feature_dict_merenv = {}, {}
    protein_2_feature_dict_merenv, protein_T5_feature_dict_merenv = {}, {}
    protein_dense_feature_dict_merenv, protein_edge_feature_dict_merenv = {}, {}

    peptide_feature_dict_merenv, peptide_ss_feature_dict_merenv = {}, {}
    peptide_2_feature_dict_merenv, peptide_T5_feature_dict_merenv = {}, {}
    peptide_dense_feature_dict_merenv, peptide_edge_feature_dict_merenv = {}, {}

    # 1. 汇总所有环境中出现的唯一序列（去重并压平嵌套列表）
    # 使用 set 去重可以极大提高效率，避免重复提取相同序列的特征
    unique_env_peps = list(set(chain.from_iterable(peptide_merenv_seq))) # 这里面存的是蛋白质序列
    unique_env_pros = list(set(chain.from_iterable(protein_merenv_seq))) # 这里面存的是多肽序列
    logging.info(unique_env_peps)

    logging.info('提取环境蛋白质特征')
    # 3. 逐个提取多肽特征
    for r_seq in unique_env_peps:
        # 包装成列表符合你底层函数的输入要求 [r_seq]
        r_feats = protein_feature_extract([r_seq], uip)
        # 存入字典（注意：r_feats 是元组，内部每一项也是列表，所以取 [0]）
        protein_feature_dict_merenv[r_seq]      = r_feats[0][0]
        protein_2_feature_dict_merenv[r_seq]    = r_feats[1][0]
        protein_dense_feature_dict_merenv[r_seq]  = r_feats[2][0]
        protein_ss_feature_dict_merenv[r_seq]     = r_feats[3][0]
        protein_T5_feature_dict_merenv[r_seq]     = r_feats[4][0]
        protein_edge_feature_dict_merenv[r_seq]   = r_feats[5][0]
        clean_uip_directory(uip)

    # 2. 逐个提取蛋白质特征 (因为底层函数一次只能处理一个序列)
    logging.info('提取环境多肽特征')
    logging.info(unique_env_pros)
    for p_seq in unique_env_pros:
        logging.info(p_seq)
        # 包装成列表 [p_seq]
        p_feats = peptide_feature_extract([p_seq], uip)
        # 存入字典
        peptide_feature_dict_merenv[p_seq]      = p_feats[0][0]
        peptide_2_feature_dict_merenv[p_seq]    = p_feats[1][0]
        peptide_dense_feature_dict_merenv[p_seq]  = p_feats[2][0]
        peptide_ss_feature_dict_merenv[p_seq]     = p_feats[3][0]
        peptide_T5_feature_dict_merenv[p_seq]     = p_feats[4][0]
        peptide_edge_feature_dict_merenv[p_seq]   = p_feats[5][0]
        logging.info('清理中间结果--环境！')
        clean_uip_directory(uip)

    peptide_merenv_seq_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_feature_dict_merenv, 800).long()
    protein_merenv_seq_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_feature_dict_merenv, 50).long()
    peptide_merenv_ss_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_ss_feature_dict_merenv, 800).long()
    protein_merenv_ss_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_ss_feature_dict_merenv, 50).long()
    peptide_merenv_2_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_2_feature_dict_merenv, 800).long()
    protein_merenv_2_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_2_feature_dict_merenv, 50).long()
    peptide_merenv_pretrain_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_T5_feature_dict_merenv, 800)
    protein_merenv_pretrain_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_T5_feature_dict_merenv, 50)
    peptide_merenv_dense_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_dense_feature_dict_merenv, 800)
    protein_merenv_dense_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_dense_feature_dict_merenv, 50)
    peptide_merenv_edge_padded = pad_and_stack_contact_maps(peptide_merenv_seq, protein_edge_feature_dict_merenv, 800)
    protein_merenv_edge_padded = pad_and_stack_contact_maps(protein_merenv_seq, peptide_edge_feature_dict_merenv, 50)

    logging.info('特征送入模型')
    ouputs_all, ouputs_all_pep, ouputs_all_pro = DCNPA_Predict(peptide_seq_feature, protein_seq_feature, peptide_ss_feature, protein_ss_feature,
                                                                peptide_2_feature, protein_2_feature, peptide_pretrain_feature, protein_pretrain_feature, 
                                                                peptide_dense_feature, protein_dense_feature, peptide_edge_feature, protein_edge_feature, 
                                                                peptide_mask, protein_mask,
                                                                peptide_simenv_seq_padded, protein_simenv_seq_padded, peptide_simenv_ss_padded, protein_simenv_ss_padded,
                                                                peptide_simenv_2_padded, protein_simenv_2_padded, peptide_simenv_pretrain_padded, protein_simenv_pretrain_padded, 
                                                                peptide_simenv_dense_padded, protein_simenv_dense_padded, peptide_simenv_edge_padded, protein_simenv_edge_padded,
                                                                peptide_sim_mask, protein_sim_mask, peptide_sim_index, protein_sim_index,
                                                                peptide_merenv_seq_padded, protein_merenv_seq_padded, peptide_merenv_ss_padded, protein_merenv_ss_padded,
                                                                peptide_merenv_2_padded, protein_merenv_2_padded, peptide_merenv_pretrain_padded, protein_merenv_pretrain_padded,
                                                                peptide_merenv_dense_padded, protein_merenv_dense_padded, peptide_merenv_edge_padded, protein_merenv_edge_padded,
                                                                pep_len, pro_len)

    # 把结果保存下来，用pickle
    result = [pep_seq_list[0], pro_seq_list[0], ouputs_all, ouputs_all_pep, ouputs_all_pro]
    result_filepath = open(os.path.join(uip, 'result.pkl'), 'wb')
    pickle.dump(result, result_filepath)
    result_filepath.close()


