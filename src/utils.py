import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import re
import RNA

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
    elif seq_type == "protein":
        alphabet = AA
    else:
        raise ValueError("seq_type must be 'nuc' or 'protein'")
    return np.array([alphabet[x] for x in sequence], dtype=int)


def get_one_hot(data, num_classes):
    N, L = data.shape
    oh = np.zeros((N, L, num_classes), dtype=float)
    oh[np.arange(N)[:, None], np.arange(L), data] = 1.0
    return oh


def get_random_vectors(data, num_classes, d):
    N, L = data.shape
    rnd = np.random.randn(num_classes, d)          # each class → N(0, I_d)
    out = rnd[data]                                # shape: (N, L, d)
    return out


def read_fasta(infile, seq_type="protein"):
    results = {}
    with open(infile, 'r') as f:
        name = None
        for l in f:
            l = l.strip()
            if l.startswith("#"):
                pass
            if l.startswith(">"):
                name = l[1:]
                results[name] = ""
            elif name:
                if seq_type == "nuc":
                    cleaned = re.sub(r'[^ACGU]', '-', l.upper())
                elif seq_type == "protein":
                    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '-', l.upper())
                else:
                    cleaned = l.upper()
                results[name] += cleaned
    return results


def read_fasta_no_insertions(infile, seq_type="protein"):
    results = {}
    with open(infile, 'r') as f:
        name = None
        for l in f:
            l = l.strip()
            if l.startswith("#"):
                continue
            if l.startswith(">"):
                name = l[1:]
                results[name] = ""
            elif name:
                # remove lowercase insertions
                l = re.sub(r'[a-z]', '', l)

                l = l.upper()
                if seq_type == "nuc":
                    cleaned = re.sub(r'[^ACGU]', '-', l)
                elif seq_type == "protein":
                    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '-', l)
                else:
                    cleaned = l

                results[name] += cleaned
    return results


def msa_to_oh(msa, seq_type="nuc", rand=False, d=4):
    if seq_type == "nuc":
        k = len(NUC)       # 5
    elif seq_type == "protein":
        k = len(AA)        # 21
    else:
        raise ValueError("seq_type must be 'nuc' or 'protein'")

    if isinstance(msa, dict):
        msa_l = [seq_to_indices(seq, seq_type) for seq in msa.values()]
    else:
        msa_l = [seq_to_indices(seq, seq_type) for seq in msa]
    msa_arr = np.stack(msa_l)              # (N, L)
    if rand:
        return get_random_vectors(msa_arr, num_classes=k, d=d)  # (N, L, k)
    else:
        return get_one_hot(msa_arr, num_classes=k)  # (N, L, k)


def oh_to_msa(msa_oh, seq_type="nuc"):
    if seq_type == "nuc":
        idx_to_sym = INDEX_TO_NUC
    elif seq_type == "protein":
        idx_to_sym = INDEX_TO_AA
    else:
        raise ValueError("seq_type must be 'nuc' or 'protein'")

    idxs = np.argmax(msa_oh, axis=-1)     # (N, L)
    return [''.join(idx_to_sym[i] for i in row) for row in idxs]


def msa_to_ind(msa, seq_type="nuc"):
    if type(msa) is dict:
        msa = msa.values()
    return np.stack([seq_to_indices(seq, seq_type) for seq in msa])


def ind_to_msa(ind_array):
    return ["".join(INDEX_TO_NUC[i] for i in row) for row in ind_array]


def get_contact(j_parms):
    F = np.sqrt(np.sum(j_parms**2, axis=(1, 3)))
    np.fill_diagonal(F, 0.0)
    Fapc = F - np.outer(F.sum(1), F.sum(0)) / F.sum()
    return Fapc


def random_onehot(N: int, L: int, k: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, k, size=(N, L))
    X = np.zeros((N, L * k), float)
    rows = np.repeat(np.arange(N), L)
    cols = symbols.ravel() + np.tile(np.arange(L) * k, N)
    X[rows, cols] = 1.0
    return X, symbols


def clustered_onehot(N: int, L: int, k: int, n_clusters: int, p_mut: float = 0.1, seed=None):
    rng = np.random.default_rng(seed)

    # Generate prototypes and expand to N sequences
    protos = rng.integers(0, k, (n_clusters, L))
    labels = rng.integers(0, n_clusters, N)
    S = protos[labels].copy()

    # Apply mutations
    mask = rng.random((N, L)) < p_mut
    S[mask] = rng.integers(0, k, mask.sum())

    # Vectorized One-Hot
    X = np.zeros((N, L * k), float)
    X[np.arange(N).repeat(L), S.ravel() + np.tile(np.arange(L) * k, N)] = 1.0

    return X, S, labels

def categorical_cov(pi):
    """ covariance of one categorical one-hot variable """
    pi = np.asarray(pi)
    return np.diag(pi) - np.outer(pi, pi)


def sample_covariance(L, k, rho=0.0, pi=None):
    """
    L   : length of sequence
    k   : alphabet size
    rho : coupling strength in [0, 1]
    pi  : base categorical distribution (default uniform)

    returns Sigma: (L*k) x (L*k) covariance matrix
    """
    if pi is None:
        pi = np.ones(k) / k

    C = categorical_cov(pi)     # single-site covariance, k x k
    I = np.eye(L)
    J = np.ones((L, L))

    Sigma = (1 - rho) * np.kron(I, C) + rho * np.kron(J, C)
    return Sigma


def sample_sequences_from_cov(Sigma, L, k, n_samples, mu=None):
    """
    Sigma: (D,D) covariance in one-hot space, D = L*k
    mu   : (D,) mean in one-hot space (if None, use zeros)
    returns: (n_samples, L) integer symbols in {0,...,k-1}
    """
    D = L * k

    if mu is None:
        mu = np.zeros(D)
    else:
        mu = np.asarray(mu)

    # eigendecomposition: Sigma = V diag(evals) V^T
    evals, V = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)          # numerical safety

    # A = V diag(sqrt(evals))  (covariance square root)
    A = V * np.sqrt(evals)[None, :]            # (D,D)

    # Gaussian samples: y = mu + A z,  z ~ N(0,I)
    z = np.random.randn(n_samples, D)          # (n_samples,D)
    Y = mu[None, :] + z @ A.T                  # (n_samples,D)

    # decode to sequences: blockwise argmax over alphabet
    Y_blocks = Y.reshape(n_samples, L, k)
    seqs = np.argmax(Y_blocks, axis=2)         # (n_samples, L)
    return seqs


def random_precision(L, k,
                     sigma_coup=0.1,   # std of off-diagonal couplings
                     x=0.5,            # extra conservation for one residue
                     edge_density=0.1,
                     eps=1e-3):
    """
    Random precision matrix Θ for an L-site, k-state model.

    - Off-diagonal site couplings are random Gaussian.
    - At each site, the diagonal block is made strictly diagonally dominant
      (Θ ≻ 0), and one residue at that site gets extra diagonal 'x'
      (more conserved than the others).
    """
    D = L * k
    Theta = np.zeros((D, D))

    # 1. random couplings between sites
    for i in range(L):
        for j in range(i + 1, L):
            if np.random.rand() < edge_density:
                J_ij = np.random.normal(0.0, sigma_coup, size=(k, k))
                Theta[i*k:(i+1)*k, j*k:(j+1)*k] = -J_ij
                Theta[j*k:(j+1)*k, i*k:(i+1)*k] = -J_ij.T

    # 2. diagonal blocks: enforce SPD + pick one more-conserved residue
    for i in range(L):
        # total magnitude of couplings touching site i
        off_block = Theta[i*k:(i+1)*k, :]
        off_sum = np.sum(np.abs(off_block))

        # base diagonal large enough for strict diagonal dominance
        base_diag = off_sum + eps

        # diagonal block: all residues get base_diag, one gets extra x
        block = base_diag * np.eye(k)
        ki = np.random.randint(k)      # residue index more conserved at site i
        block[ki, ki] += x

        Theta[i*k:(i+1)*k, i*k:(i+1)*k] = block

    return Theta


def make_mu_biased(L, k, fav, p_fav):      # favored residue probability
    pi = np.full(k, (1.0 - p_fav)/(k-1))
    pi[fav] = p_fav
    return np.tile(pi, L)


def softmax(theta):
    theta = theta - theta.max()           # numerical stability
    e = np.exp(theta)
    return e / e.sum()


def random_split(Xi, y, frac=0.8, rng=None):
    """
    Randomly split Xi and y into train/validation sets.
    Xi : numpy array
    y  : torch tensor or numpy array
    Returns:
        Xi_t, Xi_v, y_t, y_v
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(Xi)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(frac * n)

    Xi_t, Xi_v = Xi[idx[:k]], Xi[idx[k:]]
    y_t, y_v = y[idx[:k]], y[idx[k:]]

    return Xi_t, Xi_v, y_t, y_v


def metrics(pred, target):
    # to numpy
    p = pred.detach().cpu().numpy().ravel()
    t = target.detach().cpu().numpy().ravel()

    r_pearson, pp  = pearsonr(p, t)
    r_spearman, ps = spearmanr(p, t)

    ss_res = ((p - t)**2).sum()
    ss_tot = ((t - t.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot

    return r_pearson, r_spearman, r2


def random_rna(L):
    return "".join(np.random.choice(list("AUCG"), size=L))


def rna_inverse(structure, n=10):
    """
    Generate RNA sequences whose MFE structure matches `structure`.
    """
    seqs = set()
    for _ in range(n):
        seq, mfe = RNA.inverse_fold(random_rna(len(structure)), structure)
        seqs.add(seq)
    return seqs
