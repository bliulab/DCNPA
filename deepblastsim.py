# -*- coding: utf-8 -*-
import os
import sys
import pickle
import logging
import shutil
import torch
from tqdm import tqdm
from deepblast.utils import load_model
from deepblast.dataset.utils import states2alignment

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_SCORE = 0.95  # Keep the published repository's original STRICT >0.95 rule.


def configured_path(env_name, default_relative_path):
    configured = os.environ.get(env_name, os.path.join(PROJECT_ROOT, default_relative_path))
    return os.path.abspath(os.path.expanduser(configured))


def CountStr(pred_alignment, x, y):
    if not pred_alignment or not max(len(x), len(y)):
        return 0.0
    return pred_alignment.count(':') / max(len(x), len(y))


def CheckSeq(sequence):
    return sequence[:800]


def read_fasta(file_path):
    sequences = []
    with open(file_path, encoding='utf-8') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq.upper())
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq.upper())
    return sequences


def compute_similarity(x, y, model):
    """Return the same four fields as the original implementation."""
    x_use = CheckSeq(x)
    y_use = CheckSeq(y)
    if x_use == y_use:
        # The former (x, None, x, 1.0) produced an unusable downstream mask.
        return x_use, ':' * len(x_use), y_use, 1.0
    try:
        pred_alignment = model.align(x_use, y_use)
        if not pred_alignment:
            return None, None, None, 0.0
        score = CountStr(pred_alignment, x_use, y_use)
        x_aligned, y_aligned = states2alignment(pred_alignment, x_use, y_use)
        return x_aligned, pred_alignment, y_aligned, round(score, 4)
    except Exception as exc:
        LOGGER.warning('DeepBLAST alignment failed for lengths (%d,%d): %s',
                       len(x_use), len(y_use), exc)
        return None, None, None, 0.0


def _expand_local_alignment(x, y, target_aln, alignment, template_aln):
    """Map a local aligned substring back to complete query/template coordinates.

    DeepAnalysis numbers template residues from position zero. Therefore any
    unaligned N-terminal template residues must be represented as insertion
    columns ('2') before building index lists. Ambiguous repeated subwords
    cannot be mapped reliably and are explicitly rejected.
    """
    x_sub = target_aln.replace('-', '')
    y_sub = template_aln.replace('-', '')
    if not x_sub or not y_sub:
        raise ValueError('Empty aligned query/template subsequence')
    if x == x_sub and y == y_sub:
        return target_aln, alignment, template_aln
    if x.count(x_sub) != 1 or y.count(y_sub) != 1:
        raise ValueError('Local alignment cannot be uniquely located in raw sequences')
    xi, yi = x.find(x_sub), y.find(y_sub)
    x_pre, x_post = x[:xi], x[xi + len(x_sub):]
    y_pre, y_post = y[:yi], y[yi + len(y_sub):]
    new_target = x_pre + '-' * len(y_pre) + target_aln + x_post + '-' * len(y_post)
    new_template = '-' * len(x_pre) + y_pre + template_aln + '-' * len(x_post) + y_post
    new_alignment = '1' * len(x_pre) + '2' * len(y_pre) + alignment + '1' * len(x_post) + '2' * len(y_post)
    if not (len(new_target) == len(new_template) == len(new_alignment)):
        raise ValueError('Failed to reconstruct full alignment coordinates')
    return new_target, new_alignment, new_template


def _normalize_existing_cache(raw):
    """Accept canonical caches and migrate legacy 'query+template' records."""
    if not isinstance(raw, dict):
        raise ValueError('Similarity cache must contain a dictionary')
    canonical = {}
    legacy_detected = False
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError('Similarity cache keys must be sequences (strings)')
        if '+' in key and isinstance(value, (list, tuple)) and len(value) == 4:
            legacy_detected = True
            x, y = key.split('+', 1)
            x_aln, alignment, y_aln, score = value
            if alignment is None and x == y:
                alignment = ':' * len(x)
            if not isinstance(alignment, str):
                LOGGER.warning('Skipping legacy record with invalid alignment: %s', key[:40])
                continue
            rec = [x_aln, alignment, y_aln, score, y]
            canonical.setdefault(x, []).append(rec)
        elif isinstance(value, list) and (
                not value or all(isinstance(z, (list, tuple)) and len(z) == 5 for z in value)):
            canonical.setdefault(key, []).extend([list(z) for z in value])
        else:
            raise ValueError('Unrecognized similarity cache schema for key: ' + key[:50])
    # Avoid duplicate template records when both forward/reverse keys were present.
    for key, hits in canonical.items():
        unique = {}
        for rec in hits:
            unique[rec[4]] = rec
        canonical[key] = list(unique.values())
    return canonical, legacy_detected


def DeepBLAST(seq_list, uip, types):
    if types not in ('peptide', 'protein'):
        raise ValueError("types must be 'peptide' or 'protein'")
    os.makedirs(uip, exist_ok=True)
    db_name = 'peptides_all.fasta' if types == 'peptide' else 'proteins_all.fasta'
    sequence_db = os.path.join(PROJECT_ROOT, 'savefeatures', db_name)
    seq_all = list(dict.fromkeys(read_fasta(sequence_db)))
    queries = list(dict.fromkeys(s.upper() for s in seq_list))
    if any(len(x) > 800 for x in queries):
        raise ValueError('DeepBLAST target sequence exceeds 800 aa: refusing silent truncation')

    cache_path = os.path.join(uip, types + '_similarity_dict.pkl')
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            similarity, legacy_detected = _normalize_existing_cache(pickle.load(f))
        if legacy_detected:
            backup_path = cache_path + '.legacy.bak'
            if not os.path.exists(backup_path):
                shutil.copy2(cache_path, backup_path)
            LOGGER.info('Migrated legacy similarity dictionary (backup: %s)', backup_path)
    else:
        similarity = {}

    pending = [x for x in queries if x not in similarity]
    if pending:
        checkpoint_path = configured_path('DEEPBLAST_CHECKPOINT',
                     os.path.join('tools', 'DeepBLAST', 'deepblast-v3.ckpt'))
        prot_t5_path = configured_path('PROT_T5_MODEL',
                     os.path.join('tools', 'prot_t5_xl_uniref50'))
        for name, path in [('DEEPBLAST_CHECKPOINT', checkpoint_path),
                           ('PROT_T5_MODEL', prot_t5_path)]:
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing {name}: {path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(checkpoint_path, prot_t5_path,
                           alignment_mode='smith-waterman').to(device)

        for x in tqdm(pending, desc=f'DeepBLAST {types}'):
            hits = []
            for y in seq_all:
                x_aln, alignment, y_aln, score = compute_similarity(x, y, model)
                if x_aln is None or not isinstance(alignment, str) or score <= MIN_SCORE:
                    continue
                if not (len(x_aln) == len(alignment) == len(y_aln)):
                    LOGGER.warning('Discarding malformed alignment: %s / %s', x[:20], y[:20])
                    continue
                try:
                    x_aln, alignment, y_aln = _expand_local_alignment(
                        x, CheckSeq(y), x_aln, alignment, y_aln)
                except ValueError as exc:
                    LOGGER.warning('Discarding unresolvable alignment: %s / %s: %s',
                                   x[:20], y[:20], exc)
                    continue
                hits.append([x_aln, alignment, y_aln, score, y])
            similarity[x] = hits
            # Save after each query so an interrupted scan can resume.
            tmp = cache_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            LOGGER.info('%s query length=%d: %d raw hits (score > %.2f)',
                        types, len(x), len(hits), MIN_SCORE)

    # Ensure normalized/migrated cache is saved even when no new query was scanned.
    if not pending:
        tmp = cache_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
    LOGGER.info('DeepBLAST %s complete: %d query groups, %d total hits', types,
                len(queries), sum(len(similarity.get(q, [])) for q in queries))
    return cache_path


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python deepblastsim.py <input_seqs.pkl> <user_dir> peptide|protein')
    with open(sys.argv[1], 'rb') as f:
        seq_list = pickle.load(f)
    DeepBLAST(seq_list, sys.argv[2], sys.argv[3])
