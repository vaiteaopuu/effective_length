"""
"""

import numpy as np
import re

# alphabets
NUC = {'-': 0, 'A': 1, 'U': 2, 'C': 3, 'G': 4}
INDEX_TO_NUC = {v: k for k, v in NUC.items()}

AA = {'-': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8,
      'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16,
      'T': 17, 'W': 18, 'Y': 19, 'V': 20}
INDEX_TO_AA = {v: k for k, v in AA.items()}


def hamming_distance(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))


def seq_to_indices(sequence, seq_type="nuc"):
    if seq_type == "nuc":
        alphabet = NUC
    elif seq_type == "prot":
        alphabet = AA
    else:
        raise ValueError("seq_type must be 'nuc' or 'prot'")
    return np.array([alphabet[x] for x in sequence], dtype=int)


def get_one_hot(data, num_classes):
    N, L = data.shape
    oh = np.zeros((N, L, num_classes), dtype=float)
    oh[np.arange(N)[:, None], np.arange(L), data] = 1.0
    return oh


def read_fasta(infile, seq_type="prot", fmt="auto"):
    """Read an aligned FASTA / a2m / a3m file into a {name: sequence} dict.

    In a3m/a2m, lowercase residues and '.' are insertions relative to the
    query and are removed to recover the match columns. With fmt="auto" this
    is applied when the extension is .a3m/.a2m or any lowercase residue is
    present; use fmt="fasta" to keep lowercase as residues.
    """
    raw = {}
    name = None
    with open(infile, 'r') as f:
        for i, l in enumerate(f):
            l = l.strip().replace("\x00", "")
            if not l or (i == 0 and l.startswith("#")):
                continue
            if l.startswith(">"):
                name = l[1:]
                raw[name] = ""
            elif name is not None:
                raw[name] += l

    if fmt == "a3m" or (fmt == "auto" and (
            str(infile).endswith((".a3m", ".a2m"))
            or any(c.islower() for s in raw.values() for c in s))):
        raw = {n: re.sub(r'[a-z.]', '', s) for n, s in raw.items()}

    results = {}
    for n, s in raw.items():
        s = s.upper()
        if seq_type == "nuc":
            results[n] = re.sub(r'[^ACGUT]', '-', s).replace("T", "U")
        elif seq_type == "prot":
            results[n] = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '-', s)
        else:
            results[n] = s

    lengths = {len(s) for s in results.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"{infile}: sequences have unequal lengths {sorted(lengths)}; "
            "input must be aligned (for a3m files pass fmt='a3m')."
        )
    return results


def msa_to_oh(msa, seq_type="nuc"):
    if seq_type == "nuc":
        k = len(NUC)       # 5
    elif seq_type == "prot":
        k = len(AA)        # 21
    else:
        raise ValueError("seq_type must be 'nuc' or 'prot'")

    if isinstance(msa, dict):
        msa_l = [seq_to_indices(seq, seq_type) for seq in msa.values()]
    else:
        msa_l = [seq_to_indices(seq, seq_type) for seq in msa]
    msa_arr = np.stack(msa_l)              # (N, L)
    return get_one_hot(msa_arr, num_classes=k)  # (N, L, k)


def oh_to_msa(msa_oh, seq_type="nuc"):
    if seq_type == "nuc":
        idx_to_sym = INDEX_TO_NUC
    elif seq_type == "prot":
        idx_to_sym = INDEX_TO_AA
    else:
        raise ValueError("seq_type must be 'nuc' or 'prot'")

    idxs = np.argmax(msa_oh, axis=-1)     # (N, L)
    return [''.join(idx_to_sym[i] for i in row) for row in idxs]


def msa_to_ind(msa, seq_type="nuc"):
    if type(msa) is dict:
        msa = msa.values()
    return np.stack([seq_to_indices(seq, seq_type) for seq in msa])


def ind_to_msa(ind_array):
    return ["".join(INDEX_TO_NUC[i] for i in row) for row in ind_array]
