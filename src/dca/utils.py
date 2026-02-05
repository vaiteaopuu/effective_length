import numpy as np
import torch
import re


NUC = {'-': 0, 'A': 1, 'U': 2, 'C': 3, 'G': 4}  # Example mapping
INDEX_TO_NUC = {0: '-', 1: 'A', 2: 'U', 3: 'C', 4: 'G'}


def apply_apc(C):
    # Row and column means
    row_mean = np.mean(C, axis=1)
    col_mean = np.mean(C, axis=0)

    # Overall mean
    overall_mean = np.mean(C)

    # APC correction
    C_apc = C - np.outer(row_mean, col_mean) / overall_mean

    return C_apc


def dca_mf(Z, L, k, w=None, zs=True, eps=1e-4):
    N, M = Z.shape
    if w is None:
        w = np.ones(N) / N

    Zt = Z * np.sqrt(w[:, None])
    mu = Zt.sum(axis=0, keepdims=True)
    C  = Zt.T @ Zt - (mu.T @ mu)

    J = -np.linalg.inv((C + C.T)/2 + eps * np.eye(C.shape[0]))
    dca_scores = np.zeros((L, L))

    # Calculate DCA scores
    dca_scores = np.zeros((L, L))
    for i in range(L):
        for j in range(i+1, L):
            submatrix = J[i*k:(i+1)*k, j*k:(j+1)*k]
            dca_scores[i, j] = (submatrix**2).sum()**0.5
            dca_scores[j, i] = dca_scores[i, j]
    return apply_apc(dca_scores)


def paired_positions(sequence):
    "give the paired positions"
    # save open bracket in piles
    pile_reg, pile_pk = [], []
    pairs = []

    try:
        for i, sstruc in enumerate(sequence):
            if sstruc in "(":
                pile_reg += [i]
            elif sstruc == "[":
                pile_pk += [i]
            elif sstruc == ")":
                pairs += [(pile_reg.pop(), i)]
            elif sstruc == "]":
                pairs += [(pile_pk.pop(), i)]
    except:
        ""
    return pairs


def read_dca_score_dic(infile):
    "read the output of ALF-RNA"
    results = {}
    for line in open(infile):
        if not line.startswith("#"):
            posi_, posj_, nuc_i, nuc_j, nrj_ = line.strip().split()
            posi, posj = int(posi_), int(posj_)
            nrj = float(nrj_)
            if (posi, posj) not in results:
                results[(posi, posj)] = {(nuc_i, nuc_j): nrj}
            else:
                results[(posi, posj)][(nuc_i, nuc_j)] = nrj
    return results


def read_dca_from_text(infile):
    results = read_dca_score_dic(infile)
    positions = list(set([p for p, _ in results]))
    positions.sort()
    nb_pos = len(positions)
    h_parms = torch.zeros(size=(nb_pos, 5))
    for pi in positions:
        for ni, nii in NUC.items():
            h_parms[pi, nii] = results[pi, pi][ni, ni]
    j_parms = torch.zeros(size=(nb_pos, 5, nb_pos, 5))
    for pi in positions:
        for pj in positions[pi+1:]:
            for ni, nii in NUC.items():
                for nj, nji in NUC.items():
                    j_parms[pi, nii, pj, nji] = results[pi, pj][ni, nj]
    return h_parms, j_parms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hamming_distance(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))


def get_one_hot(data, num_classes=5):
    """Efficient one-hot encoding in PyTorch."""
    return torch.nn.functional.one_hot(data, num_classes=num_classes).to(dtype=torch.float32)


def seq_to_indices(sequence):
    """Convert a nucleotide sequence to index tensor."""
    return torch.tensor([NUC[nt] for nt in sequence], dtype=torch.long)


def read_fasta(infile):
    """Read a FASTA file and return a dictionary of sequences."""
    results = {}
    with open(infile, 'r') as f:
        name = None
        for l in f:
            if l.startswith(">"):
                name = l.strip()[1:]
                results[name] = ""
            else:
                cleaned = re.sub(r'[^AGUC]', '-', l.strip())
                results[name] += cleaned
    return results


def msa_to_oh(msa):
    """Convert a multiple sequence alignment (MSA) to one-hot encoding."""
    if type(msa) is dict:
        msa_l = [seq_to_indices(seq) for seq in msa.values()]
    elif type(msa) is list:
        msa_l = [seq_to_indices(seq) for seq in msa]
    msa_tensor = torch.stack(msa_l)  # Shape (num_sequences, sequence_length)
    return get_one_hot(msa_tensor)  # Shape (num_sequences, sequence_length, 5)

def oh_to_msa(msa_oh):
    """ Convert one-hot encoded MSA back to list of sequences (strings)."""
    if not torch.is_tensor(msa_oh):
        msa_oh = torch.as_tensor(msa_oh)

    # indices of max one-hot position
    idxs = msa_oh.argmax(dim=-1).squeeze(-1)  # [N, L]
    N, L = idxs.shape

    msa = []
    for n in range(N):
        seq = ''.join(INDEX_TO_NUC[idxs[n, l].item()] for l in range(L))
        msa.append(seq)
    return msa


def msa_to_ind(msa):
    """Convert a multiple sequence alignment (MSA) to one-hot encoding."""
    msa_l = [seq_to_indices(seq) for seq in msa.values()]
    return torch.stack(msa_l)  # Shape (num_sequences, sequence_length)


def set_zerosum_gauge(j_params):
    j_params -= j_params.mean(dim=1, keepdim=True) + \
        j_params.mean(dim=3, keepdim=True) - \
        j_params.mean(dim=(1, 3), keepdim=True)
    return j_params


def get_contact(j_parms):
    F = torch.sqrt(torch.square(j_parms).sum([1, 3]))
    F = F - torch.diag(F.diag())
    Fapc = F - torch.outer(F.sum(1), F.sum(0)) / F.sum()
    return Fapc


def select_by_gap(chains_oh, gap_count, max_n=0, gap_index=0):
    """ Select up to max_n sequences with exactly `gap_count` gaps. """
    # gaps per sequence (since one-hot, sum over position of the gap channel)
    counts = chains_oh[:, :, gap_index].sum(dim=1).to(torch.int64)   # [M]
    cand = (counts == int(gap_count)).nonzero(as_tuple=True)[0]      # [C]

    if cand.numel() == 0:
        return chains_oh.new_empty((0, chains_oh.size(1), chains_oh.size(2))), cand

    if max_n > 0 and cand.numel() > max_n:
        # sample without replacement
        perm = torch.randperm(cand.numel(), generator=None)
        cand = cand[perm[:max_n]]

    subset = chains_oh.index_select(0, cand)
    return subset, chains_oh
