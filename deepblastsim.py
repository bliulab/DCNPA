# -*- coding: utf-8 -*-
# %%
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
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>")

import logging
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def configured_path(env_name, default_relative_path):
    """Resolve a local model path, allowing an environment override."""
    configured = os.environ.get(
        env_name, os.path.join(PROJECT_ROOT, default_relative_path)
    )
    return os.path.abspath(os.path.expanduser(configured))

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
    try:
        if x == y:
            return x, None, x, 1.0
        
        x_use = CheckSeq(x)
        y_use = CheckSeq(y)
        
        pred_alignment = model.align(x_use, y_use)
        similarity_tmp = CountStr(pred_alignment, x_use, y_use)
        
        x_aligned, y_aligned = states2alignment(pred_alignment, x_use, y_use)
        return x_aligned, pred_alignment, y_aligned, round(similarity_tmp, 4)
    except Exception as e:
        return None, None, None, 0.0

def DeepBLAST(seq_list, uip, types):
    if types == 'peptide':
        sequence_db = os.path.join(PROJECT_ROOT, "savefeatures", "peptides_all.fasta")
    else:
        sequence_db = os.path.join(PROJECT_ROOT, "savefeatures", "proteins_all.fasta")

    seq_all = read_fasta(sequence_db)

    seq_main = seq_list

    checkpoint_path = configured_path(
        "DEEPBLAST_CHECKPOINT", os.path.join("tools", "DeepBLAST", "deepblast-v3.ckpt")
    )
    prot_t5_path = configured_path(
        "PROT_T5_MODEL", os.path.join("tools", "prot_t5_xl_uniref50")
    )
    for env_name, model_path in (
        ("DEEPBLAST_CHECKPOINT", checkpoint_path),
        ("PROT_T5_MODEL", prot_t5_path),
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Required model path not found: {model_path}. "
                f"Set {env_name} to the installed location."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        checkpoint_path,
        prot_t5_path,
        alignment_mode='smith-waterman'
    ).to(device)

    dict_name = os.path.join(uip, types+'_similarity_dict.pkl')

    if os.path.exists(dict_name):
        with open(dict_name, 'rb') as f:
            similarity = pickle.load(f)
    else:
        similarity = {}

    for x in tqdm(seq_main, desc="Overall Progress"):
        
        for y in seq_all:
            key_tmp_1 = x + '+' + y
            key_tmp_2 = y + '+' + x

            if key_tmp_1 in similarity:
                continue
                
            x_aligned, pred_alignment, y_aligned, score = compute_similarity(x, y, model)

            if x_aligned is not None and score is not None and score > 0.95:
                similarity[key_tmp_1] = [x_aligned, pred_alignment, y_aligned, score]
                
                if x_aligned == y_aligned:
                    pred_alignment_reverse = pred_alignment
                elif pred_alignment:
                    pred_alignment_reverse = (pred_alignment.replace('2', '#')
                                              .replace('1', '2')
                                              .replace('#', '1'))
                else:
                    pred_alignment_reverse = None
                
                similarity[key_tmp_2] = [y_aligned, pred_alignment_reverse, x_aligned, score]

        with open(dict_name, 'wb') as output:
            pickle.dump(similarity, output)


# %%
if __name__ == "__main__":

    tmp_seq_path = sys.argv[1]
    uip_path = sys.argv[2]
    data_type = sys.argv[3]
    
    with open(tmp_seq_path, 'rb') as f:
        seq_list = pickle.load(f)

    DeepBLAST(seq_list, uip_path, data_type)
