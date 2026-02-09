# -*- coding: utf-8 -*-
"""
Created on 2024-10-30

conda deactivate
source /home/chenshutao/Tools/DeepBLAST/deepblast/bin/activate

对蛋白质结构相似性进行提取-基于Work3的数据

@author: Shutao Chen (shutao.chen@bit.edu.cn)
"""

# %%
# 设置GPU
import os
import sys
import time
import torch
import pickle
from tqdm import tqdm
from deepblast.utils import load_model
from deepblast.dataset.utils import states2alignment

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

import warnings
# Suppress FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
# Suppress specific warning for the T5 tokenizer
warnings.filterwarnings("ignore", message="You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>")

# # 设置可见的GPU设备，假设你有多张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
logging.basicConfig(level=logging.INFO)

# %%
def CountStr(pred_alignment, x, y):
    count = 0
    if len(pred_alignment) > 0:
        for tmp in range(len(pred_alignment)):
            if pred_alignment[tmp] == ':':
                count += 1
            else:
                continue
    seq_maxlen = max(len(x), len(y))
    result = count/seq_maxlen
    return result

def CheckSeq(sequence):
    # 对于蛋白质来说
    if len(sequence) > 800:
        sequence = sequence[0:800]
    return sequence

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

# %%
def compute_similarity(x, y, model):
    """
    保持原有的逻辑，但增加了基础的错误捕获
    """
    try:
        if x == y:
            return x, None, x, 1.0
        
        x_use = CheckSeq(x)
        y_use = CheckSeq(y)
        
        # 注意：如果 model.align 依然报 Numba CUDA 错误，
        # 这一行在没有 GPU 的机器上可能依然无法运行。
        pred_alignment = model.align(x_use, y_use)
        similarity_tmp = CountStr(pred_alignment, x_use, y_use)
        
        x_aligned, y_aligned = states2alignment(pred_alignment, x_use, y_use)
        return x_aligned, pred_alignment, y_aligned, round(similarity_tmp, 4)
    except Exception as e:
        print(f"\n计算序列对失败: {x[:10]}... + {y[:10]}... | 错误: {e}")
        return None, None, None, 0.0

def DeepBLAST(seq_list, uip, types):
    # 加载数据
    if types == 'peptide':
        seq_all = read_fasta("/data/www/DCNPA/webserver/savefeatures/subset_test/peptides_subset_200.fasta")
    else:
        seq_all = read_fasta("/data/www/DCNPA/webserver/savefeatures/subset_test/proteins_subset_200.fasta")
    
    seq_all = seq_all[:10] # 测试用，后续删掉！

    seq_main = seq_list

    model = load_model("/data/www/DCNPA/webserver/tools/DeepBLAST/deepblast-v3.ckpt", 
                        "/data/www/KGIPA/webserver/tools/prot_t5_xl_uniref50",
                        alignment_mode = 'smith-waterman').cuda()


    dict_name = os.path.join(uip, types+'_similarity_dict.pkl')

    
    # 加载已有的结果（实现断点续传，防止崩溃后从头再来）
    if os.path.exists(dict_name):
        with open(dict_name, 'rb') as f:
            similarity = pickle.load(f)
        logging.info(f"已加载现有记录: {len(similarity)} 条")
    else:
        similarity = {}

    # 外层循环：遍历缺失的蛋白质
    # 使用 tqdm 显示进度：[已处理缺失蛋白数 / 总缺失数]
    for x in tqdm(seq_main, desc="Overall Progress"):
        
        # 内层循环：与所有蛋白质配对
        for y in seq_all:
            key_tmp_1 = x + '+' + y
            key_tmp_2 = y + '+' + x

            # 跳过已计算过的（双向检查）
            if key_tmp_1 in similarity:
                continue

            # 直接调用函数，不通过进程池
            x_aligned, pred_alignment, y_aligned, score = compute_similarity(x, y, model)

            # 如果计算成功（不是 None）
            if x_aligned is not None and score is not None and score > 0.95:
                # 存储正向结果
                similarity[key_tmp_1] = [x_aligned, pred_alignment, y_aligned, score]
                
                # 处理反向结果
                if x_aligned == y_aligned:
                    pred_alignment_reverse = pred_alignment
                elif pred_alignment:
                    pred_alignment_reverse = (pred_alignment.replace('2', '#')
                                              .replace('1', '2')
                                              .replace('#', '1'))
                else:
                    pred_alignment_reverse = None
                
                similarity[key_tmp_2] = [y_aligned, pred_alignment_reverse, x_aligned, score]

        # 每跑完一个 x 的所有配对，就保存一次，安全第一
        with open(dict_name, 'wb') as output:
            pickle.dump(similarity, output)

    logging.info(f"所有计算完成，最终结果共计 {len(similarity)} 条记录，保存至: {dict_name}")


# %%
if __name__ == "__main__":
    # 接收来自主程序的参数
    # sys.argv[1]: 序列列表的临时存储路径 (pkl)
    # sys.argv[2]: uip 路径
    # sys.argv[3]: types ('peptide' 或 'protein')
    
    tmp_seq_path = sys.argv[1]
    uip_path = sys.argv[2]
    data_type = sys.argv[3]
    
    # 1. 读取序列列表
    with open(tmp_seq_path, 'rb') as f:
        seq_list = pickle.load(f)
        
    # 2. 运行你原本的 DeepBLAST 函数
    DeepBLAST(seq_list, uip_path, data_type)
