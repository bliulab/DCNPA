# %%
import torch
import pickle
from tqdm import tqdm

# %%
def get_similarity_env(batch_seq, similarity_dict, feature_dict):
    """
    提取并筛选与 batch_seq 中序列相似的环境信息，
    仅保留末尾元素存在于 feature_dict 中的条目。
    """
    sim_env = [similarity_dict.get(seq, []) for seq in batch_seq]
    
    for i in range(len(sim_env) - 1, -1, -1):
        for j in range(len(sim_env[i]) - 1, -1, -1):
            if sim_env[i][j][-1] not in feature_dict:
                del sim_env[i][j]
    return sim_env

def extract_last_element_from_nested_lists(nested_list):
    """
    Extracts the last element from each inner list of a nested list.

    Args:
        nested_list (list): A list of lists, where each inner list contains elements.

    Returns:
        list: A new list containing the last element of each inner list.
    """
    return [[sublist[-1] for sublist in outer_list] for outer_list in nested_list]

def build_mask_from_peptide_simenv(sim_info):
    """
    输入为 peptide_simenv 的一条记录，返回:
    - mask tensor: shape [L_target, 1]
    - env_indices: List[int], 环境蛋白中对应的索引（-1 表示未匹配）
    
    参数:
        sim_info: list，结构为 [target_aln, alignment, env_aln, score, raw_seq]
        
    返回:
        mask: torch.FloatTensor [L_target, 1]
        env_indices: list of int
    """
    target_aln, alignment, env_aln = sim_info[0], sim_info[1], sim_info[2]

    mask_list = []
    env_indices = []
    t_idx, e_idx = 0, 0

    for t_res, a_char, e_res in zip(target_aln, alignment, env_aln):
        if t_res == '-':
            if e_res != '-':
                e_idx += 1
            continue

        if a_char == ':' and e_res != '-':
            mask_list.append(0.0)
            env_indices.append(e_idx)
        else:
            mask_list.append(1.0)
            env_indices.append(-1)

        if e_res != '-':
            e_idx += 1
        t_idx += 1

    mask = torch.tensor(mask_list, dtype=torch.float32).unsqueeze(-1)
    return mask, env_indices



# %%
def DeepAnalysis(seq_list, deepblast_dict_path, T5_feature):

    with open(deepblast_dict_path, "rb") as f:
        similarity_deepblast_dict = pickle.load(f)

    # for tmp in range(len(complex_ids)):
    for tmp in tqdm(range(len(seq_list)), desc="Processing complex_ids"):

        # 环境多肽及环境蛋白质的特征整理  - 下面这个地方在报错
        simenv = get_similarity_env(seq_list, similarity_deepblast_dict, T5_feature)
        simenv_seq = extract_last_element_from_nested_lists(simenv)[0]

        # 环境多肽及环境蛋白质对应的mask - 存储ResAlign的结果
        sim_mask, sim_indices = [], []
        for envtmp in range(len(simenv[0])):
            sim_info = simenv[0][envtmp]
            mask_tmp, indices_tmp = build_mask_from_peptide_simenv(sim_info)
            sim_mask.append(mask_tmp)
            sim_indices.append(indices_tmp)

    return simenv_seq, sim_mask, sim_indices

# %%
