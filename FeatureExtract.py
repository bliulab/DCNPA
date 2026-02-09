# -*- coding: utf-8 -*-
import sys
from utils import *
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO)

def save_to_fasta(seqlist, seqtype, uip_path):
    """
    Save the sequence list as a FASTA format file
    param seqlist: Sequence list (e.g., ['MAG...', 'KTY...'])
    param seqtype: Sequence type (e.g., 'Protein' or 'Peptide')
    param uip_path: Storage path
    """
    fasta_filename = f"{seqtype}_Seq.fasta"
    fasta_path = os.path.join(uip_path, fasta_filename)
    
    with open(fasta_path, 'w', encoding='utf-8') as f:
        for i, seq in enumerate(seqlist):
            f.write(f">{seqtype}_{i}\n")
            f.write(f"{seq}\n")
            
    print(f"Successfully saved to: {fasta_path}")

# Total protein feature extraction script
def protein_feature_extract(prolist, uip):
    protein_max_length = 800
    save_to_fasta(prolist, "Protein", uip)
    
    # Define the function task for each feature extraction
    def extract_features():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_seq_pro = executor.submit(sequence_feature_extract, prolist, protein_max_length)
            future_physical_pro = executor.submit(physical_feature_extract, prolist, protein_max_length)
            future_dense_pro = executor.submit(dense_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_ss_pro = executor.submit(secondary_structure_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_pretrain_pro = executor.submit(pretrain_feature_extract, prolist, uip, 'Protein', protein_max_length)
            future_edge_pro = executor.submit(edge_feature_extract, prolist, uip, 'Protein', protein_max_length)
            
            # Wait for all features to be extracted and obtain the results
            x_seq_pro = future_seq_pro.result()
            x_physical_pro = future_physical_pro.result()
            x_dense_pro = future_dense_pro.result()
            x_ss_pro = future_ss_pro.result()
            x_pretrain_pro = future_pretrain_pro.result()
            x_edge_pro = future_edge_pro.result()
            
        return x_seq_pro, x_physical_pro, x_dense_pro, x_ss_pro, x_pretrain_pro, x_edge_pro
    
    return extract_features()

# Total peptide feature extraction script
def peptide_feature_extract(peplist, uip):
    peptide_max_length = 50
    save_to_fasta(peplist, "Peptide", uip)

    # Define the function task for each feature extraction
    def extract_features():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_seq_pep = executor.submit(sequence_feature_extract, peplist, peptide_max_length)
            future_physical_pep = executor.submit(physical_feature_extract, peplist, peptide_max_length)
            future_dense_pep = executor.submit(dense_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_ss_pep = executor.submit(secondary_structure_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_pretrain_pep = executor.submit(pretrain_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            future_edge_pep = executor.submit(edge_feature_extract, peplist, uip, 'Peptide', peptide_max_length)
            
            # Wait for all features to be extracted and obtain the results
            x_seq_pep = future_seq_pep.result()
            x_physical_pep = future_physical_pep.result()
            x_dense_pep = future_dense_pep.result()
            x_ss_pep = future_ss_pep.result()
            x_pretrain_pep = future_pretrain_pep.result()
            x_edge_pep = future_edge_pep.result()
        
        return x_seq_pep, x_physical_pep, x_dense_pep, x_ss_pep, x_pretrain_pep, x_edge_pep
    
    return extract_features()


def parallel_feature_extract(pro_seq_list, pep_seq_list, uip, delay=30):
    """
    Execute sequentially, adding delays between tasks.
    """
    logging.info("Starting sequential feature extraction with delay")
    
    # Protein feature extraction
    logging.info("Step 1: Protein feature extraction")
    protein_features = protein_feature_extract(pro_seq_list, uip)
    logging.info(f"✅ Protein features completed, waiting {delay}s before next task")
    
    # Add delay to allow the system to recover
    time.sleep(delay)
    
    # Peptide feature extraction
    logging.info("Step 2: Peptide feature extraction")
    peptide_features = peptide_feature_extract(pep_seq_list, uip)
    logging.info("✅ Peptide features completed")
    
    return protein_features, peptide_features

