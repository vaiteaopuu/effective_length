"""
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def average_dist(X):
    N, L, k = X.shape
    X = X.reshape(N, -1)
    mat_dist = L - X @ X.T
    avg_dist = mat_dist[np.triu_indices(N, k=1)].mean()
    return avg_dist


def kmers_from_id_matrix(ids, kmer_len):
    ids = np.asarray(ids)
    N, L = ids.shape

    n_kmers = L - kmer_len + 1
    sN, sL = ids.strides

    kmers = as_strided(ids, shape=(N, n_kmers, kmer_len), strides=(sN, sL, sL))
    return kmers.copy()


def kmer_entropy(ids, kmer_len):
    kmers = kmers_from_id_matrix(ids, kmer_len)    # (N, n_kmers, kmer_len)
    flat = kmers.reshape(-1, kmer_len)            # all k-mers in one list

    unique_kmers, counts = np.unique(flat, axis=0, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-15))      # numerical safety
    return np.exp(H)


def pos_entropy(X: np.ndarray, pos: bool = False, neff: bool = False, eps: float = 1e-12):
    assert len(X.shape) == 3, "MSA must be NxLxk, where N is the number of sequences, L the length, and k is 5 or 21"
    N, L, k = X.shape
    p = X.reshape(N, -1).mean(axis=0)                 # frequency per symbol
    p = np.clip(p, eps, 1.0)           # avoid log(0)
    p = p.reshape(L, k)
    if neff:
        return np.exp((-(p * np.log(p))).sum(axis=-1).sum())
    elif pos:
        return np.exp((-(p * np.log(p))).sum(axis=-1))
    else:
        return np.exp((-(p * np.log(p))).sum(axis=-1).mean())


def neff_seq(X: np.ndarray, thres: float = 0.8, wei: bool = False):
    N, L, k = X.shape
    X = X.reshape(N, -1)
    C = X @ X.T / L

    S = (C > thres).astype(float)    # similarity matrix
    counts = S.sum(axis=1)

    w = 1.0 / counts

    if wei:
        return w / w.sum()           # normalized weights
    else:
        return w.sum()               # effective number of sequences


def helmert_basis(k: int) -> np.ndarray:
    # k x (k-1), columns span {v : 1^T v = 0}, columns orthonormal
    H = np.zeros((k, k-1), float)
    for j in range(k-1):
        H[:j+1, j] = 1.0
        H[j+1, j] = -(j+1)
        H[:j+2, j] /= np.sqrt((j+1)*(j+2))
    return H


def to_zero_sum(X: np.ndarray, L: int, k: int) -> np.ndarray:
    # X: N x (kL) one-hot, L positions, k symbols
    Q = helmert_basis(k)                 # k x (k-1)
    T = np.kron(np.eye(L), Q)            # (kL) x (L*(k-1))
    return X @ T                         # N x L*(k-1)


def weighted_covariance(Z, w):
    # Z: N×M,  w: N, sum(w)=1
    Zt = Z * np.sqrt(w[:, None])          # weighted data: diag(√w) Z
    mu = (w @ Z)                 # μ_w ∈ ℝ^{M×1}
    C  = Zt.T @ Zt - np.outer(mu, mu)          # Zᵀ diag(w) Z − μ_w μ_wᵀ
    return C, mu, Zt


def effective_length(Z, w=None, tol=1e-12, zs=True, neff=False, reg=0.):
    N, L, k = Z.shape
    Z = Z.reshape(N, -1)
    if zs:
        Z = to_zero_sum(Z, L, k)

    N, M = Z.shape
    if w is None:
        w = np.ones(N) / N

    C, mu, Zt = weighted_covariance(Z, w)

    vals = np.linalg.eigvalsh((C + C.T) / 2.0 + reg * np.eye(M))
    vals = vals[vals > tol]
    if vals.size == 0:
        return 0.0

    p = vals / vals.sum()
    H = -np.sum(p * np.log(p))

    if neff:
        return k ** (np.exp(H) / (k-1 if zs else k))
    else:
        return float(np.exp(H) / (k-1 if zs else k))


def spectral_entropy(C, tol=1e-12):
    lam_full, U_full = np.linalg.eigh(C)  # M eigenvalues, M×M eigenvectors
    mask = lam_full > tol
    lam = lam_full[mask]                  # r
    U   = U_full[:, mask]                 # M×r

    S = lam.sum()
    p = lam / S
    H = -np.sum(p * np.log(p))
    return H, lam, U
