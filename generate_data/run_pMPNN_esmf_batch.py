# Adapted from https://github.com/jinzhuwei/se3_diffusion

import os
import time
import numpy as np
import torch
import subprocess
from biotite.sequence.io import fasta
import biotite.structure.io as bsio
from omegaconf import DictConfig, OmegaConf
import esm
import sys
import pandas as pd
from tqdm import tqdm
from tmtools import tm_align
from data import utils as du
import GPUtil
from time import sleep


folder_inp = '/scratch/p287956/mpnn-study/test-mpnn'
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

def run_pMPNN_esmf_batch(
		pmpnn_dir: str,
		sequences_per_sample: int,
		global_folder: None,
		sample_ids: np.ndarray):
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
	model = esm.pretrained.esmfold_v1()
	model = model.eval().cuda()
	with torch.no_grad():
		plddts_des_dict = {}
		if global_folder is not None:
			sample_folders = [folder for folder in os.listdir(global_folder) if folder[0]!='.']
			print(sample_folders)
			for sample_id in sample_folders:
				fasta_headers = []
				fasta_seqs = [] # reset fasta_seqs and headers
				print(sample_id)
				decoy_pdb_dir = os.path.join(global_folder, sample_id) # reference path is incorrect bc im sorting by family
				reference_pdb_path = [i for i in os.listdir(decoy_pdb_dir) if i.endswith('.pdb')][0]
				print(reference_pdb_path)
				# reference_pdb_path = os.path.join(decoy_pdb_dir, sample_id+'.pdb')
				output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
				process = subprocess.Popen([
				'python',
				f'{pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
				f'--input_path={decoy_pdb_dir}',
				f'--output_path={output_path}',
				])
				
				start_pmpnn = time.time()
				_ = process.wait()
				num_tries = 0
				ret = -1
				pmpnn_args = [
				'python',
				f'{pmpnn_dir}/protein_mpnn_run.py',
				'--out_folder',
				decoy_pdb_dir,
				'--jsonl_path',
				output_path,
				'--num_seq_per_target',
				str(sequences_per_sample),
				'--sampling_temp',
				'0.1',
				'--seed',
				'38',
				'--batch_size',
				'1'
			]
				print(pmpnn_args)

				while ret < 0:
					try:
						process = subprocess.Popen(
							pmpnn_args,
							stdout=subprocess.DEVNULL,
							stderr=subprocess.STDOUT
						)
						ret = process.wait()
						print(process)
					except Exception as e:
						num_tries += 1
						# log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
						torch.cuda.empty_cache()
						if num_tries > 4:
							raise e
					
				mpnn_fasta_path = os.path.join(
					decoy_pdb_dir,
					'seqs',
					os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
				)
				print(mpnn_fasta_path)
				end_mpnn = time.time()
				print(f"ProteinMPNN took {end_mpnn-start_pmpnn:.2f} seconds")

				fasta_file = fasta.FastaFile.read(mpnn_fasta_path)
				fasta_headers.extend(list(fasta_file.keys()))
				fasta_seqs.extend(list(fasta_file.values()))

				# log.info(f"Run ESM on {len(fasta_seqs)} sequences")
				print(f"Run ESM on {len(fasta_seqs)} sequences")

				# Run ESMFold on each ProteinMPNN sequence
				all_esm_pdbs = []
				start_esmfold = time.time()
				for seq in fasta_seqs:
					output = model.infer_pdb(seq)
					all_esm_pdbs.append(output)
				end_esmfold = time.time()
				print(f"ESMFold took {end_esmfold-start_esmfold:.2f} seconds")

				mpnn_results = {
					'tm_score': [],
					'plddt': [],
					'sample_path': [],
					'header': [],
					'sequence': [],
					'rmsd': [],
					}

				for i in range(len(fasta_seqs)):
					# define header, sequence, parsed pdb
					input_pdb_loc = os.path.join(decoy_pdb_dir, reference_pdb_path)
					header = fasta_headers[i]
					sequence = fasta_seqs[i]

					esm_pdb = all_esm_pdbs[i]
					decoy_pdb_dir = os.path.join(global_folder, sample_id)
					esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
					os.makedirs(esmf_dir, exist_ok=True)

					# modify here for the sample name if i == 0 so that it has the wt name
					# put a condition where i check whether wt is present or not for self-consistency on genie and RFdiff
					if i == 0:
						esmf_sample_path = os.path.join(esmf_dir, reference_pdb_path)
						with open(esmf_sample_path, "w") as f:
							f.write(esm_pdb)
					else:
						esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
						with open(esmf_sample_path, "w") as f:
							f.write(esm_pdb)

					sample_feats = du.parse_pdb_feats('sample', input_pdb_loc)
					esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
					sample_seq = du.aatype_to_seq(sample_feats['aatype'])

					_, tm_score = calc_tm_score(
					sample_feats['bb_positions'], esmf_feats['bb_positions'],
					sample_seq, sample_seq)
					rmsd = calc_aligned_rmsd(
					sample_feats['bb_positions'], esmf_feats['bb_positions'])
					plddt = get_plddt(esmf_sample_path)

					mpnn_results['rmsd'].append(rmsd)
					mpnn_results['plddt'].append(plddt)
					mpnn_results['tm_score'].append(tm_score)
					mpnn_results['sample_path'].append(esmf_sample_path)
					mpnn_results['header'].append(header)
					mpnn_results['sequence'].append(sequence)

					if i % sequences_per_sample == 0: # 11 samples in total
						if i > 0:
						# Save results to CSV
							csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
							mpnn_results = pd.DataFrame(mpnn_results)
							mpnn_results.to_csv(csv_path)
							mpnn_results = {
							'tm_score': [],
							'plddt': [],
							'sample_path': [],
							'header': [],
							'sequence': [],
							'rmsd': [],
							}

	total_end = time.time()
	print(f"Total time: {total_end-total_start:.2f} seconds")
dict_plddts = run_pMPNN_esmf_batch(pMPNN_folder, 10, folder_inp, None)
