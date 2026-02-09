# -*- coding: utf-8 -*-
import sys
from utils import *
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO)

def save_to_fasta(seqlist, seqtype, uip_path):
    """
    将序列列表保存为 FASTA 格式文件
    :param seqlist: 序列列表 (e.g., ['MAG...', 'KTY...'])
    :param seqtype: 序列类型 (e.g., 'Protein' 或 'Peptide')
    :param uip_path: 存储路径
    """
    # 构建文件名，例如: Protein_Seq.fasta 或 Peptide_Seq.fasta
    fasta_filename = f"{seqtype}_Seq.fasta"
    fasta_path = os.path.join(uip_path, fasta_filename)
    
    with open(fasta_path, 'w', encoding='utf-8') as f:
        for i, seq in enumerate(seqlist):
            # FASTA 格式：第一行以 > 开头，紧跟 ID；第二行是序列本身
            f.write(f">{seqtype}_{i}\n")
            f.write(f"{seq}\n")
            
    print(f"已成功保存至: {fasta_path}")

# 总的蛋白质特征提取脚本
def protein_feature_extract(prolist, uip):
    # 预先设定蛋白质的最长序列长度为800
    protein_max_length = 800
    save_to_fasta(prolist, "Protein", uip)
    
    # 定义各个特征提取的函数任务
    def extract_features():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_seq_pro = executor.submit(sequence_feature_extract, prolist, protein_max_length) # 没问题
            future_physical_pro = executor.submit(physical_feature_extract, prolist, protein_max_length) # 没问题
            future_dense_pro = executor.submit(dense_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_ss_pro = executor.submit(secondary_structure_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_pretrain_pro = executor.submit(pretrain_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_edge_pro = executor.submit(edge_feature_extract, prolist, uip, 'Protein', protein_max_length)
            
            # 等待所有特征提取完成并获取结果
            x_seq_pro = future_seq_pro.result()
            x_physical_pro = future_physical_pro.result()
            x_dense_pro = future_dense_pro.result()
            x_ss_pro = future_ss_pro.result()
            x_pretrain_pro = future_pretrain_pro.result()
            x_edge_pro = future_edge_pro.result()
            
        return x_seq_pro, x_physical_pro, x_dense_pro, x_ss_pro, x_pretrain_pro, x_edge_pro
    
    return extract_features()

# 总的多肽特征提取脚本
def peptide_feature_extract(peplist, uip):
    peptide_max_length = 50
    save_to_fasta(peplist, "Peptide", uip)

    # 定义各个特征提取的函数任务
    def extract_features():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_seq_pep = executor.submit(sequence_feature_extract, peplist, peptide_max_length)
            future_physical_pep = executor.submit(physical_feature_extract, peplist, peptide_max_length)
            future_dense_pep = executor.submit(dense_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_ss_pep = executor.submit(secondary_structure_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_pretrain_pep = executor.submit(pretrain_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_edge_pep = executor.submit(edge_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            
            # 等待所有特征提取完成并获取结果
            x_seq_pep = future_seq_pep.result()
            x_physical_pep = future_physical_pep.result()
            x_dense_pep = future_dense_pep.result()
            x_ss_pep = future_ss_pep.result()
            x_pretrain_pep = future_pretrain_pep.result()
            x_edge_pep = future_edge_pep.result()
        
        return x_seq_pep, x_physical_pep, x_dense_pep, x_ss_pep, x_pretrain_pep, x_edge_pep
    
    return extract_features()

# def peptide_feature_extract(peplist, uip):
#     peptide_max_length = 50
#     save_to_fasta(peplist, "Peptide", uip)

#     def extract_features():
#         # 1. 轻量级/计算密集型任务 (容易受 GIL 影响，建议直接串行，稳如泰山)
#         logging.info("  -> 正在提取 Seq & Physical 特征...")
#         x_seq_pep = sequence_feature_extract(peplist, peptide_max_length)
#         x_physical_pep = physical_feature_extract(peplist, peptide_max_length)

#         # 2. IO/外部软件调用 (容易产生文件冲突，建议串行)
#         logging.info("  -> 正在提取 SS & Edge 特征 (外部工具)...")
#         x_ss_pep = secondary_structure_feature_extract(peplist, uip, 'Peptide', peptide_max_length)
#         x_edge_pep = edge_feature_extract(peplist, uip, 'Peptide', peptide_max_length)

#         # 3. 深度学习模型推理 (最吃资源，严禁与其他 GPU 任务并行)
#         logging.info("  -> 正在进行 T5 & Dense 模型推理 (GPU/Big RAM)...")
#         x_dense_pep = dense_feature_extract(peplist, uip, 'Peptide', peptide_max_length)
#         x_pretrain_pep = pretrain_feature_extract(peplist, uip, 'Peptide', peptide_max_length)
        
#         return x_seq_pep, x_physical_pep, x_dense_pep, x_ss_pep, x_pretrain_pep, x_edge_pep
    
#     return extract_features()


def parallel_feature_extract(pro_seq_list, pep_seq_list, uip, delay=30):
    """
    顺序执行，任务间添加延迟
    """
    logging.info("Starting sequential feature extraction with delay")
    
    # 蛋白质特征提取
    logging.info("Step 1: Protein feature extraction")
    protein_features = protein_feature_extract(pro_seq_list, uip)
    logging.info(f"✅ Protein features completed, waiting {delay}s before next task")
    
    # 添加延迟，让系统恢复
    time.sleep(delay)
    
    # 多肽特征提取
    logging.info("Step 2: Peptide feature extraction")
    peptide_features = peptide_feature_extract(pep_seq_list, uip)
    logging.info("✅ Peptide features completed")
    
    return protein_features, peptide_features

