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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    peptide_data = {
        "seq_feature": peptide_features[0],
        "2_feature": peptide_features[1],
        "dense_feature": peptide_features[2],
        "ss_feature": peptide_features[3],
        "pretrain_feature": peptide_features[4],
        "edge_feature": peptide_features[5]
    }
    
    protein_data = {
        "seq_feature": protein_features[0],
        "2_feature": protein_features[1],
        "dense_feature": protein_features[2],
        "ss_feature": protein_features[3],
        "pretrain_feature": protein_features[4],
        "edge_feature": protein_features[5]
    }

    pep_file = os.path.join(save_dir, "peptide_features_combined.pkl")
    pro_file = os.path.join(save_dir, "protein_features_combined.pkl")

    with open(pep_file, 'wb') as f:
        pickle.dump(peptide_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(pro_file, 'wb') as f:
        pickle.dump(protein_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_db_in_venv(seq_list, uip, types):
    venv_python = "/data/webserver/anaconda3/envs/DeepBLAST/bin/python"
    script_path = "deepblastsim.py"
    
    tmp_pkl = os.path.join(uip, f"input_seqs_{types}.pkl")
    with open(tmp_pkl, 'wb') as f:
        pickle.dump(seq_list, f)

    cmd = [venv_python, script_path, tmp_pkl, uip, types]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise e
    finally:
        if os.path.exists(tmp_pkl):
            os.remove(tmp_pkl)


def clean_uip_directory(uip_path):
    if not os.path.exists(uip_path):
        return

    keep_filenames = {'do_shap.txt', 'Peptide_Main.fasta', 'Protein_Main.fasta', 'Env_All.fasta'}

    for filename in os.listdir(uip_path):
        file_path = os.path.join(uip_path, filename)
        
        if os.path.isfile(file_path):
            if filename not in keep_filenames:
                try:
                    os.remove(file_path)
                except Exception as e:

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

    if not seqs_nested:
        return torch.tensor([])
    
    batch_size = len(seqs_nested)
    max_num_subseq = max(len(sublist) for sublist in seqs_nested)

    if len(feature_dict) == 0:
        result = np.zeros(
            (batch_size, max_num_subseq, max_len),
            dtype=np.float32
        )
        return torch.from_numpy(result)

    sample_feat = next(iter(feature_dict.values()))
    if sample_feat.ndim == 1:
        result = np.zeros((batch_size, max_num_subseq, max_len), dtype=np.float32)

        for i, group in enumerate(seqs_nested):
            for j, seq in enumerate(group):
                feat = feature_dict[seq]  # [L]
                length = min(len(feat), max_len)
                result[i, j, :length] = feat[:length]

        return torch.from_numpy(result)  # [B, N, L]
    elif sample_feat.ndim == 2:
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
    if not seqs_nested:
        return torch.tensor([])
    
    batch_size = len(seqs_nested)
    max_num_subseq = max(len(sublist) for sublist in seqs_nested)

    result = np.zeros((batch_size, max_num_subseq, max_len, max_len), dtype=np.float32)

    for i, group in enumerate(seqs_nested):
        for j, seq in enumerate(group):
            if seq in contact_map_dict:
                contact = contact_map_dict[seq]
                L = min(contact.shape[0], max_len)
                result[i, j, :L, :L] = contact[:L, :L]
            else:
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
    

    model = DCNPA().to(device)
    model_path = os.path.join(MODEL_FOLD, 'checkpoint.pth')
    
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

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
    
    ouputs_all = ouputs_all.detach().numpy()
    ouputs_all = ouputs_all[0, :prolen[0], :peplen[0]]  # (prolen, peplen)
    
    ouputs_all_pep = np.max(ouputs_all, 0)  # (peplen,)
    ouputs_all_pro = np.max(ouputs_all, 1)  # (prolen,)

    return ouputs_all, ouputs_all_pep, ouputs_all_pro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-uip", required=True, help="User dir")
    parser.add_argument("-do_shap", default="0", help="Whether to run SHAP (0/1)")
    args = parser.parse_args()
    uip = args.uip
    
    pep_uip = os.path.join(uip, 'Peptide_Main.fasta')
    pro_uip = os.path.join(uip, 'Protein_Main.fasta')
    env_uip = os.path.join(uip, 'Env_All.fasta')
    
    pep_seq_list, pro_seq_list = [], []
    for i in open(pep_uip):
        if i[0] != '>':
            pep_seq_list.append(i.strip().upper())
    for i in open(pro_uip):
        if i[0] != '>':
            pro_seq_list.append(i.strip().upper())

    (protein_features, peptide_features) = parallel_feature_extract(pro_seq_list, pep_seq_list, uip)

    clean_uip_directory(uip)

    protein_seq_feature, protein_2_feature, protein_dense_feature, protein_ss_feature, protein_pretrain_feature, protein_edge_feature = protein_features
    peptide_seq_feature, peptide_2_feature, peptide_dense_feature, peptide_ss_feature, peptide_pretrain_feature, peptide_edge_feature = peptide_features

    pep_len = [len(pep_seq_list[0])]
    pro_len = [len(pro_seq_list[0])]

    peptide_mask = SeqMask(pep_len, 50, device)  # torch.Size([16, 50])
    protein_mask = SeqMask(pro_len, 800, device)  # torch.Size([16, 800])
    
    do_shap = args.do_shap.lower() in ["1", "true", "yes"]
    
    if do_shap:
        run_db_in_venv(pep_seq_list, uip, 'peptide')
        run_db_in_venv(pro_seq_list, uip, 'protein')

        base_path = 'savefeatures/'
        prefixes = ['peptide', 'protein']
        feat_mapping = {
            '': 'feature', 
            'ss_': 'ss_feature', 
            '2_': '2_feature', 
            'T5_': 'T5_feature', 
            'dense_': 'dense_feature', 
            'edge_': 'edge_feature'
        }

        feats = {}
        for pref in prefixes:
            for ft, label in feat_mapping.items():
                file_name = f"{pref}_{label}_dict.pkl"
                file_path = os.path.join(base_path, file_name)
                
                dict_key = f"{pref}_{label}"
                with open(file_path, 'rb') as f:
                    feats[dict_key] = pickle.load(f, encoding='iso-8859-1')

        peptide_simenv_seq, peptide_sim_mask, peptide_sim_index = DeepAnalysis(pep_seq_list, os.path.join(uip, 'peptide_similarity_dict.pkl'), feats['peptide_T5_feature'])
        protein_simenv_seq, protein_sim_mask, protein_sim_index = DeepAnalysis(pro_seq_list, os.path.join(uip, 'protein_similarity_dict.pkl'), feats['protein_T5_feature'])

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
    else:
        peptide_simenv_seq, protein_simenv_seq = [[]], [[]]
        peptide_sim_mask, peptide_sim_index, protein_sim_mask, protein_sim_index = [], [], [], []

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



    # =========================== multimer-aware dynamic context sub-network ===========================
    peptide_merenv_seq = [[]]
    protein_merenv_seq = [[]]
    
    if os.path.exists(env_uip):
        with open(env_uip, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('>'):
                    continue
                
                seq = line.upper()
                if len(seq) <= 50:
                    protein_merenv_seq[0].append(seq)
                else:
                    peptide_merenv_seq[0].append(seq)

    protein_feature_dict_merenv, protein_ss_feature_dict_merenv = {}, {}
    protein_2_feature_dict_merenv, protein_T5_feature_dict_merenv = {}, {}
    protein_dense_feature_dict_merenv, protein_edge_feature_dict_merenv = {}, {}
    peptide_feature_dict_merenv, peptide_ss_feature_dict_merenv = {}, {}
    peptide_2_feature_dict_merenv, peptide_T5_feature_dict_merenv = {}, {}
    peptide_dense_feature_dict_merenv, peptide_edge_feature_dict_merenv = {}, {}

    # Summarize unique sequences appearing in all environments (remove duplicates and flatten nested lists).
    unique_env_peps = list(set(chain.from_iterable(peptide_merenv_seq)))
    unique_env_pros = list(set(chain.from_iterable(protein_merenv_seq)))

    # Extract peptide features one by one (lower-level functions can only process one sequence at a time).
    for r_seq in unique_env_peps:
        r_feats = protein_feature_extract([r_seq], uip)
        protein_feature_dict_merenv[r_seq]      = r_feats[0][0]
        protein_2_feature_dict_merenv[r_seq]    = r_feats[1][0]
        protein_dense_feature_dict_merenv[r_seq]  = r_feats[2][0]
        protein_ss_feature_dict_merenv[r_seq]     = r_feats[3][0]
        protein_T5_feature_dict_merenv[r_seq]     = r_feats[4][0]
        protein_edge_feature_dict_merenv[r_seq]   = r_feats[5][0]
        clean_uip_directory(uip)

    # Extract protein features one by one (lower-level functions can only process one sequence at a time).
    for p_seq in unique_env_pros:
        p_feats = peptide_feature_extract([p_seq], uip)
        peptide_feature_dict_merenv[p_seq]      = p_feats[0][0]
        peptide_2_feature_dict_merenv[p_seq]    = p_feats[1][0]
        peptide_dense_feature_dict_merenv[p_seq]  = p_feats[2][0]
        peptide_ss_feature_dict_merenv[p_seq]     = p_feats[3][0]
        peptide_T5_feature_dict_merenv[p_seq]     = p_feats[4][0]
        peptide_edge_feature_dict_merenv[p_seq]   = p_feats[5][0]
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

    
    # DCNPA prediction
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

    # Save results to pkl file
    result = [pep_seq_list[0], pro_seq_list[0], ouputs_all, ouputs_all_pep, ouputs_all_pro]
    result_filepath = open(os.path.join(uip, 'result.pkl'), 'wb')
    pickle.dump(result, result_filepath)
    result_filepath.close()
