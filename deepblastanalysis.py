import logging
import pickle
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def get_similarity_env(batch_seq, similarity_dict, feature_dict):
    """Read canonical query->list-of-5-field-hits; filter missing features.

    Never mutate the cache in memory, as other queries may share records.
    """
    if not isinstance(similarity_dict, dict):
        raise ValueError('Expected a dictionary keyed by target sequences')
    sim_env = []
    for seq in batch_seq:
        candidates = similarity_dict.get(seq, [])
        if not isinstance(candidates, (list, tuple)):
            raise ValueError(f'Expected a list of templates for {seq[:25]}')
        selected = []
        missing = 0
        for record in candidates:
            if not isinstance(record, (list, tuple)) or len(record) != 5:
                raise ValueError('Incorrect DeepBLAST record format; expected '
                                 '[target_aln, alignment, template_aln, score, raw_template_seq]')
            target_aln, alignment, template_aln, score, raw_template_seq = record
            if raw_template_seq not in feature_dict:
                missing += 1
                continue
            if not all(isinstance(s, str) for s in (target_aln, alignment, template_aln)):
                raise ValueError('DeepBLAST alignment components must be strings')
            if not (len(target_aln) == len(alignment) == len(template_aln)):
                raise ValueError('DeepBLAST aligned strings have inconsistent lengths')
            # Original mask construction assumes all target residues are represented.
            if target_aln.replace('-', '') != seq:
                LOGGER.warning('Skipping partial or mismatched query alignment: %s / %s',
                               seq[:20], raw_template_seq[:20])
                continue
            # Similarly, template indices are relative to the raw template sequence.
            if template_aln.replace('-', '') != raw_template_seq[:len(template_aln.replace('-', ''))]:
                LOGGER.warning('Skipping template alignment without resolvable raw coordinates: %s',
                               raw_template_seq[:20])
                continue
            selected.append(list(record))
        LOGGER.info('DeepAnalysis query=%s... retrieved=%d usable=%d missing_features=%d',
                    seq[:16], len(candidates), len(selected), missing)
        sim_env.append(selected)
    return sim_env


def extract_last_element_from_nested_lists(nested_list):
    return [[record[-1] for record in group] for group in nested_list]


def build_mask_from_peptide_simenv(sim_info):
    target_aln, alignment, env_aln = sim_info[0], sim_info[1], sim_info[2]
    mask_list = []
    env_indices = []
    e_idx = 0
    for t_res, a_char, e_res in zip(target_aln, alignment, env_aln):
        if t_res == '-':
            if e_res != '-':
                e_idx += 1
            continue
        if a_char == ':' and e_res != '-':
            mask_list.append(0.0)  # 0: use template feature (code convention)
            env_indices.append(e_idx)
        else:
            mask_list.append(1.0)  # 1: use target feature
            env_indices.append(-1)
        if e_res != '-':
            e_idx += 1
    return torch.tensor(mask_list, dtype=torch.float32).unsqueeze(-1), env_indices


def DeepAnalysis(seq_list, deepblast_dict_path, T5_feature):
    with open(deepblast_dict_path, 'rb') as f:
        similarity_deepblast_dict = pickle.load(f)
    simenv = get_similarity_env(seq_list, similarity_deepblast_dict, T5_feature)
    simenv_seq = extract_last_element_from_nested_lists(simenv)
    all_sim_masks, all_sim_indices = [], []
    for sequence_env in tqdm(simenv, desc='Processing DeepBLAST templates'):
        masks, indices = [], []
        for sim_info in sequence_env:
            mask_tmp, indices_tmp = build_mask_from_peptide_simenv(sim_info)
            masks.append(mask_tmp)
            indices.append(indices_tmp)
        all_sim_masks.append(masks)
        all_sim_indices.append(indices)
    return simenv_seq, all_sim_masks, all_sim_indices
