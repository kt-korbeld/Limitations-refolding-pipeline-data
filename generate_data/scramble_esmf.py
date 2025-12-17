import os
import time
import json
import gzip
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import logging
from typing import Optional

from biotite.sequence import ProteinSequence as ps
from biotite.sequence.io import fasta
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import biotite.structure.io as bsio

import mdtraj as md
from tqdm import tqdm

from data import utils as du
from experiments import utils as eu
from analysis import metrics

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# TODO: output this shit into a fasta file and then run esmf on the same structure of the folders except replacing self-consistency for scramble

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

def get_structure_tite(pdb_loc:str, compressed=False):
    if compressed:
        with gzip.open(pdb_loc, "rt") as file_handle:
            pdb_file = pdb_io.PDBFile.read(file_handle)
    else:
        with open(pdb_loc, "r") as file_handle:
            pdb_file = pdb_io.PDBFile.read(file_handle)
    protein = pdb_io.get_structure(pdb_file, model=1, extra_fields=['b_factor'])
    protein = protein[(protein.hetero == False) & (protein.ins_code == '')]
    return protein

def get_chain(df:pd.DataFrame, pdb_loc:str):
    pdb_name = pdb_loc.split('/')[-1].split('.')[0][3:]
    chain_id = (df[df['pdb_id'] == pdb_name]['chain']).values[0]
    protein = get_structure_tite(pdb_loc)
    protein_chain = protein[protein.chain_id == chain_id]
    return protein_chain

def get_sequence(protein:struc.AtomArray):
    seq = protein[(protein.atom_name == 'CA')&(protein.hetero==False)].res_name
    one_letter = list(ps(seq))
    return one_letter

def scramble_sequence(seq_list:str, 
                      num_seqs:int, 
                      folder_out:str, 
                      pct_scr_list:list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
                      seed:int=213):
    '''
    Randomly shuffles the sequences of amino acids provided as a list for a given percentage of positions and writes num_seqs outputs into a .fa format file
    Returns:
        out_seqs: list of scrambled sequences len = num_seqs
    '''
    np.random.seed(seed)
    for pct_scr in pct_scr_list:
        out_seqs = []
        out_seqs.append(''.join(seq_list))
        num_positions = int(len(seq_list)*pct_scr)
        for i in range(num_seqs):
            scramble_positions = np.random.choice(len(seq_list), num_positions, replace=False)
            scrambled_seq = seq_list.copy()
            for j in scramble_positions:
                scrambled_seq[j] = np.random.choice(list(alphabet))
            out_seqs.append(''.join(scrambled_seq))

        folder_save = os.path.join(folder_out, f'scramble', f'scrambled_{pct_scr}')
        if not os.path.exists(folder_save):
            os.makedirs(folder_save, exist_ok=True)
        
        with open(os.path.join(folder_save, f'scrambled_{pct_scr}.fa'), 'w') as f:
            for i in range(len(out_seqs)):
                if i == 0: # write the original sequence
                    f.write(f'>native\n{out_seqs[i]}\n')
                else:
                    f.write(f'>scrambled_{i}\n{out_seqs[i]}\n')
    return None

def make_scramble(folder:str, num_samples:int, pct_scr_list:list=[0.1, 0.2, 0.4, 0.5, 0.6], seed:int=213, csv_subset:Optional[pd.DataFrame]=None):
    '''
    Iterates over a folder of pdb files and does the following:
    1. Extracts the sequence of amino acids
    2. Randomly shuffles the sequences of amino acids for a given percentage of positions
    3. Writes num_samples outputs into a .fa format file
    '''
    protein_folders = [i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))]
    np.random.seed(seed)
    if csv_subset is not None:
        print(f'Filtering pdbs by the subset provided')
        pdbs = csv_subset['pdb_id'].values
        protein_folders = [i for i in protein_folders if i in pdbs]
        print(f'Scrambling sequences for {len(protein_folders)} pdbs')
    for pdb_name in tqdm(protein_folders, desc=f'Scrambling sequences for each pdb in {folder}'):
        pdb_loc = os.path.join(folder, pdb_name, f'{pdb_name}.pdb')
        protein = get_structure_tite(pdb_loc)
        seq_chain = get_sequence(protein)
        scrambled_seqs = scramble_sequence(seq_list=seq_chain, num_seqs=num_samples, folder_out=os.path.join(folder, pdb_name), pct_scr_list=pct_scr_list, seed=seed)
    return None

def get_plddt(loc):
    struct = bsio.load_structure(loc, extra_fields=["b_factor"])
    plddt = round(struct.b_factor.mean(),2)
    return plddt

def get_sse(loc):
    t = md.load(loc)
    dssp = md.compute_dssp(t)
    dssp = dssp[0]
    pct_helix = np.sum(dssp == 'H') / len(dssp)
    pct_strand = np.sum(dssp == 'E') / len(dssp)
    pct_coil = np.sum(dssp == 'C') / len(dssp)
    return pct_helix, pct_strand, pct_coil

def run_self_consistency_batch(
        sequences_per_sample: int,
        folding_model,
        proteins_dir: str,
        samples_chains: dict,
        calc_non_coil_rmsd: bool = False,
        max_res_per_esm_batch: int = 1500,
    ):
    """Run self-consistency on design proteins against reference protein."""

    # Clear GPU memory
    torch.cuda.empty_cache()
    chains = np.array(samples_chains['chain'])
    sample_ids = np.array(samples_chains['sample'])

    with torch.no_grad():
        for sample_id in tqdm(sample_ids, desc="Processing samples"):
            sample_dir = os.path.join(proteins_dir, f'{sample_id}')
            decoy_pdb_dir = os.path.join(sample_dir, 'scramble')
            if not os.path.exists(decoy_pdb_dir):
                log.error(f"No scrambled pctg found for {sample_id}")
                continue
            
            reference_pdb_path = os.path.join(sample_dir, f'{sample_id}.pdb')
            shutil.copy(reference_pdb_path, decoy_pdb_dir)
            
            # Identify and process each scrambled folder for the sample
            scrambled_pctgs = [folder for folder in os.listdir(decoy_pdb_dir) if folder.startswith('scrambled')]
            if len(scrambled_pctgs) == 0:
                log.error(f"No scrambled pctgs found for {sample_id}")
                continue

            start_time = time.time()
            for pctg in scrambled_pctgs:
                pctg_dir = os.path.join(decoy_pdb_dir, pctg)
                scrambled_fasta_path = os.path.join(pctg_dir, f'{pctg}.fa')

                if not os.path.exists(scrambled_fasta_path):
                    log.error(f"Missing fasta file for {scrambled_fasta_path}")
                    break
                
                fasta_file = fasta.FastaFile.read(scrambled_fasta_path)
                fasta_headers = list(fasta_file.keys())
                fasta_seqs = list(fasta_file.values())

                log.info(f"Run ESM on {len(fasta_seqs)} sequences in {pctg}")
                
                # Split sequences into batches
                seq_lens = [len(seq) for seq in fasta_seqs]
                
                batches = []
                batch = []
                batch_len = 0
                for i, seq_len in enumerate(seq_lens):
                    if batch_len + seq_len > max_res_per_esm_batch:
                        batches.append(batch)
                        batch = []
                        batch_len = 0
                    batch.append(fasta_seqs[i])
                    batch_len += seq_len
                if len(batch) > 0:
                    batches.append(batch)
                log.info(f"Run ESM on batches of size {[len(b) for b in batches]} with {len(batches)} batches")

                if folding_model == 'cuda':
                    import esm
                    from esm.esmfold.v1.esmfold import ESMFold
                    folding_model = esm.pretrained.esmfold_v1().eval()
                    folding_model = folding_model.cuda()
                
                all_esm_pdbs = []
                for batch in tqdm(batches, desc=f"Running ESMFold for {pctg} for {sample_id}"):
                    torch.cuda.empty_cache()
                    try:
                        all_esm_output = folding_model.infer(batch)
                        all_esm_pdbs.extend(folding_model.output_to_pdb(all_esm_output))
                    except Exception as e:
                        log.error(f"ESMFold failed for batch: {e}")
                        continue
                
                # save_results_and_metrics(
                #     all_esm_pdbs, fasta_headers, fasta_seqs, sample_id, pctg_dir,
                #     decoy_pdb_dir, reference_pdb_path, chains, sequences_per_sample,
                #     calc_non_coil_rmsd, pctg, sample_ids
                # )

                mpnn_results = {
                'tm_score': [], 
                'sample_path': [], 
                'header': [], 
                'sequence': [],
                'rmsd': [], 
                'plddt': [], 
                'pct_helix': [], 
                'pct_strand': [], 
                'pct_coil': []
            }
        
                for i in range(len(fasta_seqs)):

                    header = fasta_headers[i]
                    sequence = fasta_seqs[i]
                    esm_pdb = all_esm_pdbs[i]

                    esmf_dir = os.path.join(pctg_dir, 'esmf')
                    os.makedirs(esmf_dir, exist_ok=True)
                    log.info(i)

                    if i == 0:
                        log.info(f"Saving ESMFold output for sample {sample_id}")
                        esmf_sample_path = os.path.join(esmf_dir, f'esmf_{sample_id}.pdb')
                        with open(esmf_sample_path, "w") as f:
                            f.write(esm_pdb)
                    else:
                        esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
                        with open(esmf_sample_path, "w") as f:
                            f.write(esm_pdb)

                    chain_id = chains[np.where(sample_ids == sample_id)[0][0]]

                    sample_feats = du.parse_pdb_feats('sample', reference_pdb_path, chain_id=chain_id)
                    esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path, calc_dssp=calc_non_coil_rmsd)
                    assert sample_feats['bb_positions'].shape[0] == esmf_feats['bb_positions'].shape[0], 'Number of residues do not match'
                    sample_seq = du.aatype_to_seq(sample_feats['aatype'])
                    
                    try:
                        _, tm_score = metrics.calc_tm_score(
                            sample_feats['bb_positions'], esmf_feats['bb_positions'],
                            sample_seq, sample_seq)
                    except Exception as e:
                        log.error(f"TM Score calculation failed: {e}")
                        continue 

                    rmsd = metrics.calc_aligned_rmsd(
                        sample_feats['bb_positions'], esmf_feats['bb_positions'])
                    plddt = get_plddt(esmf_sample_path)
                    helix, strand, coil = get_sse(esmf_sample_path)

                    mpnn_results['rmsd'].append(rmsd)
                    mpnn_results['tm_score'].append(tm_score)
                    mpnn_results['sample_path'].append(esmf_sample_path)
                    mpnn_results['header'].append(header)
                    mpnn_results['sequence'].append(sequence)
                    mpnn_results['plddt'].append(plddt)
                    mpnn_results['pct_helix'].append(helix)
                    mpnn_results['pct_strand'].append(strand)
                    mpnn_results['pct_coil'].append(coil)

                    if i % sequences_per_sample == 0:
                        if i > 0:
                            # Save results to CSV
                            log.info(f"Saving results to CSV for sample {sample_id}")
                            csv_path = os.path.join(pctg_dir, 'sc_results.csv')
                            mpnn_results = pd.DataFrame(mpnn_results)
                            mpnn_results.to_csv(csv_path)
                            mpnn_results = {k: [] for k in mpnn_results}
                            log.info(f"Finished self-consistency for sample {sample_id}, resetting results")

            end_time = time.time()
            compute_time = end_time - start_time

            benchmark_data = {
                    "sample_id": sample_id,
                    "num_scrambles": len(scrambled_pctgs),
                    "num_seqs": len(fasta_seqs),
                    "sequence_length": seq_len,
                    "compute_time": compute_time
                }
                
            benchmark_path = os.path.join(decoy_pdb_dir, f'{sample_id}_benchmark.json')
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark_data, f, indent=4)
        log.info(f"Benchmark for {sample_id}: Total Time={compute_time:.2f}s, Seq_len={seq_len}, Folders={len(scrambled_pctgs)}")


def save_results_and_metrics(
        all_esm_pdbs, fasta_headers, fasta_seqs, sample_id, pctg_dir, 
        decoy_pdb_dir, reference_pdb_path, chains, sequences_per_sample, calc_non_coil_rmsd, pctg, sample_ids):
    
    mpnn_results = {
        'tm_score': [], 
        'sample_path': [], 
        'header': [], 
        'sequence': [],
        'rmsd': [], 
        'plddt': [], 
        'pct_helix': [], 
        'pct_strand': [], 
        'pct_coil': []
    }
    
    for i, (header, sequence, esm_pdb) in enumerate(zip(fasta_headers, fasta_seqs, all_esm_pdbs)):
        esmf_dir = os.path.join(pctg_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        log.info(i)

        if i == 0:
            log.info(f"Saving ESMFold output for sample {sample_id}")
            esmf_sample_path = os.path.join(esmf_dir, f'esmf_{sample_id}.pdb')
            with open(esmf_sample_path, "w") as f:
                f.write(esm_pdb)
        else:
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            with open(esmf_sample_path, "w") as f:
                f.write(esm_pdb)

        chain_id = chains[np.where(sample_ids == sample_id)[0][0]]

        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path, chain_id=chain_id)
        esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path, calc_dssp=calc_non_coil_rmsd)
        assert sample_feats['bb_positions'].shape[0] == esmf_feats['bb_positions'].shape[0], 'Number of residues do not match'
        sample_seq = du.aatype_to_seq(sample_feats['aatype'])
        
        try:
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
        except Exception as e:
            log.error(f"TM Score calculation failed: {e}")
            continue 

        rmsd = metrics.calc_aligned_rmsd(
            sample_feats['bb_positions'], esmf_feats['bb_positions'])
        plddt = get_plddt(esmf_sample_path)
        helix, strand, coil = get_sse(esmf_sample_path)

        mpnn_results['rmsd'].append(rmsd)
        mpnn_results['tm_score'].append(tm_score)
        mpnn_results['sample_path'].append(esmf_sample_path)
        mpnn_results['header'].append(header)
        mpnn_results['sequence'].append(sequence)
        mpnn_results['plddt'].append(plddt)
        mpnn_results['pct_helix'].append(helix)
        mpnn_results['pct_strand'].append(strand)
        mpnn_results['pct_coil'].append(coil)

        if i % sequences_per_sample == 0:
            if i > 0:
                # Save results to CSV
                log.info(f"Saving results to CSV for sample {sample_id}")
                csv_path = os.path.join(pctg_dir, 'sc_results.csv')
                mpnn_results = pd.DataFrame(mpnn_results)
                mpnn_results.to_csv(csv_path)
            log.info(f"Finished self-consistency for sample {sample_id}, resetting results")
            mpnn_results = {k: [] for k in mpnn_results}

def parse_args():
    parser = argparse.ArgumentParser(description="Run self-consistency batch process for designed proteins.")
    
    parser.add_argument('--proteins_dir', type=str, required=True, help="Directory where designed protein files are stored.")
    parser.add_argument('--proteins_csv', type=str, required=True, help="Path to the SCOPe CSV file.")
    parser.add_argument('--sequences_per_sample', type=int, default=8, help="Number of sequences per sample.")
    parser.add_argument('--folding_model', type=str, default='cuda', help="Folding model to use ('cuda' or other options if available).")
    parser.add_argument('--max_res_per_esm_batch', type=int, default=1500, help="Maximum number of residues per ESMFold batch.")
    parser.add_argument('--calc_non_coil_rmsd', default = False, action='store_true', help="Calculate non-coil RMSD if set.")
    return parser.parse_args()

if __name__ == "__main__": 
    
    args = parse_args()
    # Load sample data from SCOPe CSV
    df_scope40 = pd.read_csv(args.proteins_csv, index_col=0)
    pdb_ids_scope40 = df_scope40['pdb_id'].values
    pdbs_dir = args.proteins_dir

    # Prepare sample and chain lists
    samples = np.array([i for i in os.listdir(pdbs_dir) if os.path.isdir(os.path.join(pdbs_dir, i))])
    samples = np.intersect1d(samples, pdb_ids_scope40)
    log.info(f'Filtered after intersection with SCOPe40: {len(samples)} samples')

    samples_chains = {
        'sample': [],
        'chain': []
    }

    log.info(samples)
    for sample in tqdm(samples, desc='Extracting chain information'):
        protein = get_structure_tite(os.path.join(pdbs_dir, sample, f'{sample}.pdb'))
        chain = protein.chain_id[0]
        samples_chains['sample'].append(sample)
        samples_chains['chain'].append(chain)

    make_scramble(folder=pdbs_dir, num_samples=args.sequences_per_sample, seed=213, csv_subset=df_scope40)

    # Run the self-consistency batch
    run_self_consistency_batch(
        proteins_dir=args.proteins_dir,
        sequences_per_sample=args.sequences_per_sample,
        samples_chains=samples_chains,
        folding_model=args.folding_model,
        calc_non_coil_rmsd=args.calc_non_coil_rmsd, 
        max_res_per_esm_batch=args.max_res_per_esm_batch
    )
