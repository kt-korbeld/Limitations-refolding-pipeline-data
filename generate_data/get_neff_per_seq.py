import numpy as np
import pandas as pd


def readfa_seqs(fa_in):
    with open(fa_in, 'r') as f:
        lines = f.readlines()
    seqs = [i.strip('\n') for i in lines if not i[0] == '>']
    lenseq = len(seqs[0])
    seqs = [i[:lenseq] for i in seqs]
    return np.array(seqs)

def msa_encode_decode(msa_in, mode='encode'):
    if mode == 'encode':
        nr = [*range(0,21), 20] # give code from 0 to 20, count X and - as 20
        aa = 'RHKDESTNQCGPAVILMFYWX-'
        map = {i:j for i,j in zip(aa, nr)}
        msa_out = np.array([[map[i] for i in seq.upper()] for seq in msa_in])
    elif mode == 'decode':
        nr = [*range(0,21)] # give code from 0 to 20, just use -
        aa = 'RHKDESTNQCGPAVILMFYW-'
        map = {j:i for i,j in zip(aa, nr)}
        msa_out = np.array([''.join([map[i] for i in seq]) for seq in msa_in])
    else:
        msa_out = msa_in
    return msa_out

def compute_per_residue_neff(msa, identity_threshold=0.8):
    """
    Compute per-residue Neff (AlphaFold2-style).
    
    Args:
        msa (ndarray of shape (N, L)): 
            MSA as integer-encoded array. 
            Use a unique code for '-' (gap). For example: A=0,...,20, '-'=21.
        identity_threshold (float): 
            Sequence identity threshold (default 0.8).
    
    Returns:
        per_res_neff (ndarray of shape (L,)): Neff per column
        median_neff (float): median Neff across positions
    """
    N, L = msa.shape
    gap_token = msa.max()  # assume last code is gap (e.g. 21)

    per_res_neff = np.zeros(L)

    for col in range(L):
        print(col)
        # Select sequences that are not gaps at this column
        mask = msa[:, col] != gap_token
        sub_msa = msa[mask]

        if sub_msa.shape[0] == 0:
            per_res_neff[col] = 0
            continue

        # Compute pairwise identity (only on overlapping non-gap positions)
        # Build mask of non-gaps for each sequence
        nongap_masks = sub_msa != gap_token
        matches = (sub_msa[:, None, :] == sub_msa[None, :, :]) & \
                  (nongap_masks[:, None, :] & nongap_masks[None, :, :])

        # Count identity per pair (fraction over compared positions)
        overlap_counts = nongap_masks[:, None, :].sum(-1) + nongap_masks[None, :, :].sum(-1) - \
                         (nongap_masks[:, None, :] & nongap_masks[None, :, :]).sum(-1)
        overlap_counts = np.maximum(overlap_counts, 1)  # avoid division by 0

        identities = matches.sum(-1) / overlap_counts

        # Sequence weights: 1 / (number of neighbors ≥ threshold)
        neighbor_counts = (identities >= identity_threshold).sum(-1)
        weights = 1.0 / neighbor_counts

        per_res_neff[col] = weights.sum()

    median_neff = np.median(per_res_neff)
    return median_neff



msas = [os.path.join('./{}'.format(dir_in), i) for i in os.listdir(dir_in)]
msanames = [i.split('/')[-1].rsplit('.',1)[0] for i in msas]
msadep = []
neff_vals = []

for i, msa in enumerate(msas):
    print(msa)
    msa_in = readfa_seqs(msa)
    msadep.append(len(msa_in))
    msa_enc = msa_encode_decode(msa_in)
    neff = compute_per_residue_neff(msa_enc)
    neff_vals.append(neff)
    df = pd.DataFrame({'name':msanames[:i+1], 'path':msas[:i+1], 'msadep':msadep, 'neff_res_med':neff_vals})
    df.to_csv('df_neff_{}.csv'.format(dir_in))
