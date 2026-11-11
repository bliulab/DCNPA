# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import csv
import torch
import pickle
import argparse
import subprocess
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
    venv_python = os.path.expanduser(
        os.environ.get("DEEPBLAST_PYTHON", sys.executable)
    )
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deepblastsim.py"
    )

    if not os.path.isfile(venv_python):
        raise FileNotFoundError(
            f"DeepBLAST Python executable not found: {venv_python}. "
            "Set DEEPBLAST_PYTHON to the Python executable in the "
            "environment where DeepBLAST is installed."
        )
    
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


def clean_uip_directory(uip_path, preserved_names):
    """Clean only intermediates created by the just-completed extraction.

    ``preserved_names`` must be a snapshot of the directory before feature
    extraction. This avoids deleting arbitrary pre-existing user files while
    clearing new temporary outputs that can otherwise contaminate later runs.
    """
    if not os.path.isdir(uip_path):
        return
    for filename in os.listdir(uip_path):
        if filename in preserved_names:
            continue
        file_path = os.path.join(uip_path, filename)
        if os.path.isfile(file_path) and not os.path.islink(file_path):
            try:
                os.remove(file_path)
            except OSError as exc:
                logging.warning('Could not remove generated temporary file %s: %s',
                                file_path, exc)


def read_single_fasta(file_path, description):
    """Read exactly one complete (possibly multi-line) FASTA sequence."""
    from Bio import SeqIO

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{description} FASTA not found: {file_path}')

    with open(file_path, 'r', encoding='utf-8') as handle:
        records = list(SeqIO.parse(handle, 'fasta'))

    if len(records) != 1:
        raise ValueError(
            f'{description} FASTA must contain exactly one sequence; '
            f'found {len(records)} in {file_path}. '
            'Run each peptide-protein pair separately.'
        )

    sequence = str(records[0].seq).strip().upper()
    if not sequence:
        raise ValueError(f'{description} FASTA contains an empty sequence: {file_path}')
    return sequence


def read_environment_fasta(file_path):
    """Read complete environmental chains, including multi-line FASTA records."""
    from Bio import SeqIO
    if not os.path.isfile(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as handle:
        records = list(SeqIO.parse(handle, 'fasta'))
    sequences = [str(rec.seq).strip().upper() for rec in records]
    if any(not seq for seq in sequences):
        raise ValueError(f'Environment FASTA contains an empty sequence: {file_path}')
    return sequences


def SeqMask(seqlen_list, seqlen_lim, device):
    seq_mask = torch.zeros((len(seqlen_list), seqlen_lim), dtype=torch.float32)
    for i, length in enumerate(seqlen_list):
        seq_mask[i, :length] = 1.0
    seq_mask = seq_mask.to(device)
    return seq_mask


def get_nested_max_seq_len(seqs_nested, fallback_len):
    """Return the real maximum sequence length in a nested sequence list.

    If the nested list contains no sequence, use ``fallback_len`` so that
    downstream empty-context tensors still have a valid residue dimension.
    """
    lengths = [len(seq) for group in seqs_nested for seq in group]
    return max(lengths) if lengths else fallback_len

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


def pad_alignment_data(mask_groups, index_groups, max_len):
    """Pad residue-alignment metadata to tensors shaped [B, N, L]."""
    batch_size = len(mask_groups)
    max_num_env = max((len(group) for group in mask_groups), default=0)
    masks = torch.ones((batch_size, max_num_env, max_len), dtype=torch.float32)
    indices = torch.full(
        (batch_size, max_num_env, max_len), -2, dtype=torch.long
    )

    for batch_index, (batch_masks, batch_indices) in enumerate(
        zip(mask_groups, index_groups)
    ):
        for env_index, (mask, env_indices) in enumerate(
            zip(batch_masks, batch_indices)
        ):
            mask_values = torch.as_tensor(mask, dtype=torch.float32).reshape(-1)
            index_values = torch.as_tensor(env_indices, dtype=torch.long).reshape(-1)
            length = min(mask_values.numel(), index_values.numel(), max_len)
            masks[batch_index, env_index, :length] = mask_values[:length]
            indices[batch_index, env_index, :length] = index_values[:length]

    return masks, indices


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
    parser.add_argument("-use_deepblast", default="0", help="Whether to use DeepBLAST (0/1)")
    args = parser.parse_args()
    uip = args.uip
    
    pep_uip = os.path.join(uip, 'Peptide_Main.fasta')
    pro_uip = os.path.join(uip, 'Protein_Main.fasta')
    env_uip = os.path.join(uip, 'Env_All.fasta')
    
    # One peptide-protein pair per run; multi-line FASTA is supported.
    peptide_sequence = read_single_fasta(pep_uip, 'Peptide')
    protein_sequence = read_single_fasta(pro_uip, 'Protein')
    if len(peptide_sequence) > 50:
        raise ValueError('Peptide sequence exceeds the supported 50 residues.')
    if len(protein_sequence) > 800:
        raise ValueError('Protein sequence exceeds the supported 800 residues.')
    pep_seq_list = [peptide_sequence]
    pro_seq_list = [protein_sequence]

    files_before_extraction = set(os.listdir(uip))
    (protein_features, peptide_features) = parallel_feature_extract(pro_seq_list, pep_seq_list, uip)

    clean_uip_directory(uip, files_before_extraction)

    protein_seq_feature, protein_2_feature, protein_dense_feature, protein_ss_feature, protein_pretrain_feature, protein_edge_feature = protein_features
    peptide_seq_feature, peptide_2_feature, peptide_dense_feature, peptide_ss_feature, peptide_pretrain_feature, peptide_edge_feature = peptide_features

    pep_len = [len(pep_seq_list[0])]
    pro_len = [len(pro_seq_list[0])]

    # Protein-related residue dimensions are dynamic during inference.
    # The peptide branch remains unchanged at the training-time limit of 50 aa.
    protein_target_len = max(pro_len)

    peptide_mask = SeqMask(pep_len, 50, device)
    protein_mask = SeqMask(pro_len, protein_target_len, device)

    logging.info("Peptide length: %d aa", pep_len[0])
    logging.info("Protein length used for inference: %d aa", protein_target_len)
    
    use_deepblast = args.use_deepblast.lower() in ["1", "true", "yes"]
    
    if use_deepblast:
        run_db_in_venv(pep_seq_list, uip, 'peptide')
        run_db_in_venv(pro_seq_list, uip, 'protein')

        # Prefer repository-local template dictionaries; optional external path.
        base_path = os.path.abspath(os.path.expanduser(
            os.environ.get('DCNPA_FEATURES_DIR', os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'savefeatures'))))
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

        # Keep only reference templates whose complete feature set is available.
        # DeepAnalysis receives the same target->template-list schema as the
        # offline dictionaries used during model development.
        required_labels = ('feature', 'ss_feature', '2_feature', 'T5_feature',
                           'dense_feature', 'edge_feature')

        def complete_template_features(prefix):
            feature_keys = [set(feats[f'{prefix}_{label}']) for label in required_labels]
            usable = set.intersection(*feature_keys)
            logging.info('%s feature-complete template sequences: %d', prefix, len(usable))
            return {seq: feats[f'{prefix}_T5_feature'][seq] for seq in usable}

        peptide_simenv_seq, peptide_sim_masks, peptide_sim_indices = DeepAnalysis(
            pep_seq_list, os.path.join(uip, 'peptide_similarity_dict.pkl'),
            complete_template_features('peptide'))
        protein_simenv_seq, protein_sim_masks, protein_sim_indices = DeepAnalysis(
            pro_seq_list, os.path.join(uip, 'protein_similarity_dict.pkl'),
            complete_template_features('protein'))

        pep_hits = [len(group) for group in peptide_simenv_seq]
        pro_hits = [len(group) for group in protein_simenv_seq]
        logging.info('Template integration: peptide hits per query=%s; protein hits per query=%s',
                     pep_hits, pro_hits)
        if not any(pep_hits) and not any(pro_hits):
            logging.warning('No feature-complete DeepBLAST templates were retained. '
                            'Target-adaptive contributions may be absent for this input. '
                            'Check the alignment threshold and reference feature dictionaries.')

        peptide_sim_mask, peptide_sim_index = pad_alignment_data(
            peptide_sim_masks, peptide_sim_indices, 50
        )
        protein_sim_mask, protein_sim_index = pad_alignment_data(
            protein_sim_masks, protein_sim_indices, protein_target_len
        )

        peptide_simenv_seq_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_feature'], 50).long()
        protein_simenv_seq_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_feature'], protein_target_len).long()
        peptide_simenv_ss_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_ss_feature'], 50).long()
        protein_simenv_ss_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_ss_feature'], protein_target_len).long()
        peptide_simenv_2_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_2_feature'], 50).long()
        protein_simenv_2_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_2_feature'], protein_target_len).long()
        peptide_simenv_pretrain_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_T5_feature'], 50)
        protein_simenv_pretrain_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_T5_feature'], protein_target_len)
        peptide_simenv_dense_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, feats['peptide_dense_feature'], 50)
        protein_simenv_dense_padded = pad_and_stack_nested_tensor(protein_simenv_seq, feats['protein_dense_feature'], protein_target_len)
        peptide_simenv_edge_padded = pad_and_stack_contact_maps(peptide_simenv_seq, feats['peptide_edge_feature'], 50)
        protein_simenv_edge_padded = pad_and_stack_contact_maps(protein_simenv_seq, feats['protein_edge_feature'], protein_target_len)
    else:
        peptide_simenv_seq, protein_simenv_seq = [[]], [[]]
        peptide_sim_mask, peptide_sim_index, protein_sim_mask, protein_sim_index = [], [], [], []

        peptide_simenv_seq_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50).long()
        protein_simenv_seq_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, protein_target_len).long()
        peptide_simenv_ss_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50).long()
        protein_simenv_ss_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, protein_target_len).long()
        peptide_simenv_2_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50).long()
        protein_simenv_2_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, protein_target_len).long()
        peptide_simenv_pretrain_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50)
        protein_simenv_pretrain_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, protein_target_len)
        peptide_simenv_dense_padded = pad_and_stack_nested_tensor(peptide_simenv_seq, {}, 50)
        protein_simenv_dense_padded = pad_and_stack_nested_tensor(protein_simenv_seq, {}, protein_target_len)
        peptide_simenv_edge_padded = pad_and_stack_contact_maps(peptide_simenv_seq, {}, 50)
        protein_simenv_edge_padded = pad_and_stack_contact_maps(protein_simenv_seq, {}, protein_target_len)



    # =========================== multimer-aware dynamic context sub-network ===========================
    peptide_merenv_seq = [[]]
    protein_merenv_seq = [[]]
    
    for seq in read_environment_fasta(env_uip):
        if len(seq) <= 50:
            protein_merenv_seq[0].append(seq)
        else:
            peptide_merenv_seq[0].append(seq)

    # ``peptide_merenv_seq`` stores protein-like environmental chains (>50 aa)
    # in the original naming convention. Pad these chains to their real maximum
    # length instead of truncating them to a fixed protein length.
    protein_merenv_max_len = get_nested_max_seq_len(
        peptide_merenv_seq,
        fallback_len=protein_target_len
    )
    logging.info(
        "Maximum protein-like environmental-chain length: %d aa",
        protein_merenv_max_len
    )

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
        files_before_extraction = set(os.listdir(uip))
        r_feats = protein_feature_extract([r_seq], uip)
        protein_feature_dict_merenv[r_seq]      = r_feats[0][0]
        protein_2_feature_dict_merenv[r_seq]    = r_feats[1][0]
        protein_dense_feature_dict_merenv[r_seq]  = r_feats[2][0]
        protein_ss_feature_dict_merenv[r_seq]     = r_feats[3][0]
        protein_T5_feature_dict_merenv[r_seq]     = r_feats[4][0]
        protein_edge_feature_dict_merenv[r_seq]   = r_feats[5][0]
        clean_uip_directory(uip, files_before_extraction)

    # Extract protein features one by one (lower-level functions can only process one sequence at a time).
    for p_seq in unique_env_pros:
        files_before_extraction = set(os.listdir(uip))
        p_feats = peptide_feature_extract([p_seq], uip)
        peptide_feature_dict_merenv[p_seq]      = p_feats[0][0]
        peptide_2_feature_dict_merenv[p_seq]    = p_feats[1][0]
        peptide_dense_feature_dict_merenv[p_seq]  = p_feats[2][0]
        peptide_ss_feature_dict_merenv[p_seq]     = p_feats[3][0]
        peptide_T5_feature_dict_merenv[p_seq]     = p_feats[4][0]
        peptide_edge_feature_dict_merenv[p_seq]   = p_feats[5][0]
        clean_uip_directory(uip, files_before_extraction)

    peptide_merenv_seq_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_feature_dict_merenv, protein_merenv_max_len).long()
    protein_merenv_seq_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_feature_dict_merenv, 50).long()
    peptide_merenv_ss_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_ss_feature_dict_merenv, protein_merenv_max_len).long()
    protein_merenv_ss_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_ss_feature_dict_merenv, 50).long()
    peptide_merenv_2_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_2_feature_dict_merenv, protein_merenv_max_len).long()
    protein_merenv_2_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_2_feature_dict_merenv, 50).long()
    peptide_merenv_pretrain_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_T5_feature_dict_merenv, protein_merenv_max_len)
    protein_merenv_pretrain_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_T5_feature_dict_merenv, 50)
    peptide_merenv_dense_padded = pad_and_stack_nested_tensor(peptide_merenv_seq, protein_dense_feature_dict_merenv, protein_merenv_max_len)
    protein_merenv_dense_padded = pad_and_stack_nested_tensor(protein_merenv_seq, peptide_dense_feature_dict_merenv, 50)
    peptide_merenv_edge_padded = pad_and_stack_contact_maps(peptide_merenv_seq, protein_edge_feature_dict_merenv, protein_merenv_max_len)
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
