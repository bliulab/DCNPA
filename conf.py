import os

CON_FOLD = 'DCNPA/'
MODEL_FOLD = os.path.join(CON_FOLD, 'savemodel')  # 要保留

MAX_NUM_OF_SEQUENCES = 1

# 2024-10-23增加的内容
IUPRED2A_FOLD = 'tools/iupred2a'
BLAST_FOLD = 'tools/ncbi-blast-2.13.0+/bin/psiblast'
BLAST_DB_FOLD = 'tools/nrdb90/nrdb90'
BLOSUM62_FOLD = 'tools/blosum62.txt'
SCRATCH_FOLD = 'tools/SCRATCH-1D_1.2'
TRROSETTAX_FOLD = 'tools/trRosettaX'
T5MODEL_FOLD = 'tools/prot_t5_xl_uniref50'
