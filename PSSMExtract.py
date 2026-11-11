# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import tempfile
import numpy as np
from Bio import SeqIO

from conf import *

LOGGER = logging.getLogger(__name__)


def Read_SeqID(FilePath):
    """Return FASTA descriptions and full sequences (including wrapped FASTA)."""
    with open(FilePath, encoding='utf-8') as handle:
        records = list(SeqIO.parse(handle, 'fasta'))
    return [rec.description for rec in records], [str(rec.seq).upper() for rec in records]


def get_protein_blosum(proteinseq):
    blosum62 = {}
    blosum_reader = open(BLOSUM62_FOLD, 'r')
    count = 0
    for line in blosum_reader:
        count = count + 1
        if count <= 7:
            continue
        line = line.strip('\r').split()
        blosum62[line[0]] = [float(x) for x in line[1:21]]
        
    protein_lst = []
    for aa in proteinseq:
        aa = aa.upper()
        if aa not in blosum62.keys():
            aa = 'X'
        protein_lst.append(blosum62[aa])
    return np.asarray(protein_lst)

def read_pssm(pssm_file):
    """Parse PSI-BLAST ASCII PSSM residues and their original 20 score columns."""
    residues = []
    rows = []
    positions = []
    with open(pssm_file, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 22 or not parts[0].isdigit():
                continue
            if len(parts[1]) != 1 or not parts[1].isalpha():
                continue
            try:
                values = [float(x) for x in parts[2:22]]
            except ValueError:
                continue
            positions.append(int(parts[0]))
            residues.append(parts[1].upper())
            rows.append(values)

    if not positions or positions != list(range(1, len(positions) + 1)):
        raise ValueError(f'Incomplete or invalid ASCII PSSM: {pssm_file}')
    return residues, np.asarray(rows, dtype=np.float64)


def _matching_pssm(pssm_file, expected_sequence):
    """Return PSSM scores only when they belong to the exact query protein."""
    if not os.path.isfile(pssm_file):
        return None
    try:
        residues, matrix = read_pssm(pssm_file)
    except (OSError, ValueError, IndexError) as exc:
        LOGGER.warning('PSSM unreadable (%s): %s', pssm_file, exc)
        return None
    if ''.join(residues) != expected_sequence or matrix.shape != (len(expected_sequence), 20):
        LOGGER.warning('Cached PSSM does not match current protein; will regenerate: %s', pssm_file)
        return None
    return matrix


def run_simple_search(fd, output_path=None):
    """Run PSI-BLAST for one FASTA; return success status, not cached results."""
    input_fasta = os.path.abspath(fd)
    if output_path is None:
        output_path = os.path.splitext(input_fasta)[0] + '.pssm'
    cmd = [
        str(BLAST_FOLD),
        '-query', input_fasta,
        '-db', str(BLAST_DB_FOLD),
        '-evalue', '0.001',
        '-num_iterations', '10',
        '-outfmt', '5',
        '-out_ascii_pssm', output_path,
        '-num_threads', '1',
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, check=False)
    except (OSError, ValueError) as exc:
        LOGGER.warning('PSI-BLAST could not be started: %s', exc)
        return False
    if result.returncode != 0:
        LOGGER.warning('PSI-BLAST failed (exit %d): %s',
                       result.returncode, (result.stderr or '').strip()[-500:])
        return False
    return True


def generateMSA(file_path):
    """Use a verified cached profile or regenerate it without trusting stale data."""
    _, sequences = Read_SeqID(file_path)
    if len(sequences) != 1 or not sequences[0]:
        raise ValueError('Protein_Seq.fasta must contain exactly one nonempty protein sequence.')
    sequence = sequences[0]
    pssm_path = os.path.join(os.path.dirname(os.path.abspath(file_path)), 'Protein_Seq.pssm')

    if _matching_pssm(pssm_path, sequence) is not None:
        LOGGER.info('Reusing sequence-validated PSSM for protein length %d', len(sequence))
        return pssm_path

    # Write into an isolated temporary path.  Do not overwrite an existing
    # cache until the new PSSM has been validated against the current protein.
    fd, tmp_path = tempfile.mkstemp(prefix='.Protein_Seq_', suffix='.pssm',
                                    dir=os.path.dirname(os.path.abspath(file_path)))
    os.close(fd)
    os.remove(tmp_path)
    try:
        if run_simple_search(file_path, output_path=tmp_path):
            if _matching_pssm(tmp_path, sequence) is not None:
                os.replace(tmp_path, pssm_path)
                return pssm_path
            LOGGER.warning('PSI-BLAST output PSSM is invalid for current sequence.')
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    return None


def pssm_feature_extract(prolist, uip):
    """Extract one protein profile; never associate another protein's PSSM."""
    file_path = os.path.join(uip, 'Protein_Seq.fasta')
    _, sequences = Read_SeqID(file_path)
    if len(sequences) != 1:
        raise ValueError('PSSMExtract supports exactly one protein sequence per call.')
    sequence = sequences[0]
    if not sequence:
        raise ValueError('Protein sequence must not be empty.')

    if prolist is not None:
        if len(prolist) != 1 or str(prolist[0]).upper() != sequence:
            raise ValueError('Protein sequence and Protein_Seq.fasta are inconsistent; '
                             'multi-sequence PSSM reuse is not supported.')

    pssm_path = generateMSA(file_path)
    scores = _matching_pssm(pssm_path, sequence) if pssm_path else None
    if scores is None:
        LOGGER.warning('No valid PSI-BLAST PSSM; using original BLOSUM62 fallback.')
        scores = get_protein_blosum(sequence)
    return {sequence: scores}
