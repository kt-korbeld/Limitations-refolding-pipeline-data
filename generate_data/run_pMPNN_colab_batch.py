# Adapted from https://github.com/jinzhuwei/se3_diffusion

import os
import time
import numpy as np
import torch
import subprocess
from biotite.sequence.io import fasta
import biotite.structure.io as bsio
from omegaconf import DictConfig, OmegaConf
import sys
import pandas as pd
from tqdm import tqdm
from tmtools import tm_align
from data import utils as du
import GPUtil
from time import sleep


folder_inp = '/scratch/p287956/mpnn-study/test-mpnn-2'
pMPNN_folder = '/scratch/p287956/mpnn-study/ProteinMPNN-main'

def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2

def calc_aligned_rmsd(pos_1, pos_2):
    '''
    took from https://github.com/jasonkyuyim/se3_diffusion/blob/master/analysis/metrics.py
    '''
    aligned_pos_1 = du.rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))

def get_plddt(loc):
    struct = bsio.load_structure(loc, extra_fields=["b_factor"])
    plddt = round(struct.b_factor.mean(),2)
    return plddt

def gen_fa_singleseq(fasta, outpath):
    '''
    Generates single-sequence fasta files of the designs from pMPNN from the output FASTA file.
    These files can be used for single-sequence AF2 prediction (batch).
        Parameters:
            None
        Returns:
            None
    '''
    with open(fasta) as fasta_input:
        fasta_filename = os.path.basename(fasta)
        lines = fasta_input.readlines()
        all_seqs = [lines[i].strip('\n') for i in range(len(lines)) if i %2 != 0]
        count = 0
        for sequence in all_seqs:
            with open(os.path.join(outpath, '{}_{}.a3m'.format(count, fasta_filename.rsplit('.', 1)[0] )), 'w') as single_seq:
                single_seq.write('>{}_{}'.format(fasta_filename.rsplit('.',1)[0], count)+'\n'+str(sequence))
            count+=1

def try_subprocess(args_in, max_tries=4):
    """
    initiates a subprocess and tries to rerun it max_tries times
    if it fails, clearing the pytorch cache between each try.
    """
    num_tries = 0
    ret = -1
    print(args_in)
    while ret < 0:
        try:
            process = subprocess.Popen(args_in,
                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            ret = process.wait()
            print(process)
        except Exception as e:
            num_tries += 1
            torch.cuda.empty_cache()
            if num_tries > max_tries:
                raise e

def run_pMPNN_esmf_batch(pmpnn_dir: str,sequences_per_sample: int,global_folder: None, sample_ids: np.ndarray):
    """Run pMPNN and predict structures of sequences in ESMfold.
    Args:
    global_folder: directory where subfolders containing proteins (.pdb) are stored.
    decoy_pdb_dir: directory where designed protein files are stored.
    reference_pdb_path: path to reference protein file
        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
    """
    total_start = time.time()
    with torch.no_grad():
        plddts_des_dict = {}
        if global_folder is not None:
            sample_folders = [folder for folder in os.listdir(global_folder) if folder[0]!='.']
            print(sample_folders, flush=True)
            for sample_id in sample_folders:
                fasta_headers = []
                fasta_seqs = [] # reset fasta_seqs and headers
                print(sample_id, flush=True)
                decoy_pdb_dir = os.path.join(global_folder, sample_id) # reference path is incorrect bc im sorting by family
                reference_pdb_path = [i for i in os.listdir(decoy_pdb_dir) if i.endswith('.pdb')][0]
                print(reference_pdb_path, flush=True)
                # reference_pdb_path = os.path.join(decoy_pdb_dir, sample_id+'.pdb')
                output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
                process = subprocess.Popen(['python',
                                            f'{pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
                                            f'--input_path={decoy_pdb_dir}',
                                            f'--output_path={output_path}'])
                _ = process.wait()
                start_pmpnn = time.time()
                pmpnn_args = ['python',
                              f'{pmpnn_dir}/protein_mpnn_run.py',
                               '--out_folder', decoy_pdb_dir,
                                '--jsonl_path', output_path,
                                '--num_seq_per_target', str(sequences_per_sample),
                                '--sampling_temp','0.1',
                                '--seed','38',
                                '--batch_size', '1']
                try_subprocess(pmpnn_args)
                mpnn_fasta_path = os.path.join(decoy_pdb_dir, 'seqs', os.path.basename(reference_pdb_path).replace('.pdb', '.fa'))
                seqs_path = os.path.join(decoy_pdb_dir, 'seqs')
                print(mpnn_fasta_path, flush=True)
                end_mpnn = time.time()
                print(f"ProteinMPNN took {end_mpnn-start_pmpnn:.2f} seconds", flush=True)
                # read in fasta file headers and seqs
                fasta_file = fasta.FastaFile.read(mpnn_fasta_path)
                fasta_headers.extend(list(fasta_file.keys()))
                fasta_seqs.extend(list(fasta_file.values()))
                # prepare .a3m files and directory for Colabfold
                gen_fa_singleseq(mpnn_fasta_path, seqs_path)
                os.makedirs(os.path.join(seqs_path, 'fasta'), exist_ok=True)
                os.rename(mpnn_fasta_path, os.path.join(seqs_path, 'fasta', os.path.basename(mpnn_fasta_path)))
                decoy_pdb_dir = os.path.join(global_folder, sample_id)
                input_pdb_loc = os.path.join(decoy_pdb_dir, reference_pdb_path)
                colab_dir = os.path.join(decoy_pdb_dir, 'colab')
                os.makedirs(colab_dir, exist_ok=True)
                # Run Colabfold on each ProteinMPNN sequence
                start_colab = time.time()
                colab_args = ["colabfold_batch",
                              "--num-models", "1",
                              "--num-recycle", "3",
                              "--msa-mode", "single_sequence",
                              seqs_path, colab_dir]
                try_subprocess(colab_args)
                end_colab = time.time()
                print(f"Colabfold took {end_colab-start_colab:.2f} seconds", flush=True)

                mpnn_results = {'tm_score': [], 'plddt': [],'sample_path': [],'header': [],'sequence': [],'rmsd': [] }
                pdbs = sorted([i for i in os.listdir(colab_dir) if i[-4:] == '.pdb'])
                for i, pdb in enumerate(pdbs):
                    header = fasta_headers[i]
                    sequence = fasta_seqs[i]
                    # rename pdb files
                    if pdb.split('_', 1)[0] == '0':
                        colab_sample_path = os.path.join(colab_dir, reference_pdb_path)
                        os.rename(os.path.join(colab_dir, pdb), colab_sample_path)
                    else:
                       colab_sample_path = os.path.join(colab_dir, f'sample_{i}.pdb')
                       os.rename(os.path.join(colab_dir, pdb), colab_sample_path)
                    # read in renamed pdb file
                    with open(colab_sample_path, 'r') as f:
                       colab_pdb = f.read()
                    try:
                        # analyse pdb file
                        sample_feats = du.parse_pdb_feats('sample', input_pdb_loc)
                        colab_feats = du.parse_pdb_feats('folded_sample', colab_sample_path)
                        sample_seq = du.aatype_to_seq(sample_feats['aatype'])
                        _, tm_score = calc_tm_score(sample_feats['bb_positions'], colab_feats['bb_positions'],
                                                    sample_seq, sample_seq)
                        rmsd = calc_aligned_rmsd(sample_feats['bb_positions'], colab_feats['bb_positions'])
                        plddt = get_plddt(colab_sample_path)

                        mpnn_results['rmsd'].append(rmsd)
                        mpnn_results['plddt'].append(plddt)
                        mpnn_results['tm_score'].append(tm_score)
                        mpnn_results['sample_path'].append(colab_sample_path)
                        mpnn_results['header'].append(header)
                        mpnn_results['sequence'].append(sequence)
                        if i % sequences_per_sample == 0: # 11 samples in total
                            if i > 0:
                                # Save results to CSV
                                csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
                                mpnn_results = pd.DataFrame(mpnn_results)
                                mpnn_results.to_csv(csv_path)
                                mpnn_results = {
                                'tm_score': [],'plddt': [],'sample_path': [],
                                'header': [],'sequence': [],'rmsd': [] }
                    except:
                        continue 
    total_end = time.time()
    print(f"Total time: {total_end-total_start:.2f} seconds", flush=True)
dict_plddts = run_pMPNN_esmf_batch(pMPNN_folder, 10, folder_inp, None)
