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
    if len(s1) != len(s2):
        raise ValueError("sequences must have the same length")
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
    oh = np.zeros((N, L, num_classes), dtype=np.float32)
    oh[np.arange(N)[:, None], np.arange(L), data] = 1.0
    return oh


def _sanitize(seq, seq_type):
    s = seq.upper().replace(".", "-").replace("~", "-")
    if seq_type == "nuc":
        return re.sub(r'[^ACGUT-]', '-', s).replace("T", "U")
    if seq_type == "prot":
        return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY-]', '-', s)
    return s


def _check_lengths(results, infile, hint=""):
    lengths = {len(s) for s in results.values()}
    if len(lengths) > 1 or lengths in ({0}, set()):
        raise ValueError(
            f"{infile}: sequences have unequal lengths {sorted(lengths)}; "
            f"input must be aligned.{hint}"
        )
    return results


def read_fasta(infile, seq_type="prot", fmt="auto"):
    """Read an aligned FASTA / a2m / a3m file into a {name: sequence} dict.

    In a3m/a2m, lowercase residues and '.' are insertions relative to the
    query and are removed to recover the match columns. With fmt="auto" this
    is applied when the extension is .a3m/.a2m, or when lowercase is present
    and the sequences are not already the same length; use fmt="fasta" to
    keep lowercase as residues.
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
                while name in raw:          # keep duplicate headers distinct
                    name += "'"
                raw[name] = ""
            elif name is not None:
                raw[name] += l

    if fmt == "a3m" or (fmt == "auto" and (
            str(infile).endswith((".a3m", ".a2m"))
            or (any(c.islower() for s in raw.values() for c in s)
                and len({len(s) for s in raw.values()}) > 1))):
        raw = {n: re.sub(r'[a-z.]', '', s) for n, s in raw.items()}

    results = {n: _sanitize(s, seq_type) for n, s in raw.items()}
    return _check_lengths(results, infile,
                          " For a3m files pass fmt='a3m'.")


def read_stockholm(infile, seq_type="prot", fmt="auto"):
    """Read a Stockholm alignment into a {name: sequence} dict.

    Blocks are interleaved, so a repeated name is a continuation of the same
    sequence rather than a new one. hmmalign-style inserts (lowercase
    residues and '.') are removed to recover the match columns; with
    fmt="auto" this is done only when the sequences are not already the same
    length, use fmt="sto" to keep lowercase as residues.
    """
    raw = {}
    order = []
    with open(infile, 'r') as f:
        for l in f:
            l = l.strip().replace("\x00", "")
            if not l or l.startswith("#") or l.startswith("//"):
                continue
            part = l.split(None, 1)
            if len(part) != 2:
                continue
            name, seq = part[0], part[1].replace(" ", "")
            if name not in raw:
                raw[name] = ""
                order.append(name)
            raw[name] += seq

    if fmt == "a3m" or (fmt == "auto"
                        and any(c.islower() for s in raw.values() for c in s)
                        and len({len(s) for s in raw.values()}) > 1):
        raw = {n: re.sub(r'[a-z.]', '', s) for n, s in raw.items()}

    results = {n: _sanitize(raw[n], seq_type) for n in order}
    return _check_lengths(results, infile)


def read_msa(infile, seq_type="prot", fmt="auto"):
    """Read a FASTA/a2m/a3m or Stockholm alignment, chosen by extension."""
    if str(infile).lower().endswith((".sto", ".stk", ".stockholm")):
        return read_stockholm(infile, seq_type=seq_type, fmt=fmt)
    return read_fasta(infile, seq_type=seq_type, fmt=fmt)


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
    if isinstance(msa, dict):
        msa = msa.values()
    return np.stack([seq_to_indices(seq, seq_type) for seq in msa])


def ind_to_msa(ind_array, seq_type="nuc"):
    if seq_type == "nuc":
        idx_to_sym = INDEX_TO_NUC
    elif seq_type == "prot":
        idx_to_sym = INDEX_TO_AA
    else:
        raise ValueError("seq_type must be 'nuc' or 'prot'")
    return ["".join(idx_to_sym[i] for i in row) for row in ind_array]
