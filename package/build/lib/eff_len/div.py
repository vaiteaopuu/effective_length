"""
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from .utils import msa_to_oh


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


def pos_entropy(X: np.ndarray, pos: bool = False, leff: bool=False, neff: bool = False, eps: float = 1e-12):
    assert len(X.shape) == 3, "MSA must be NxLxk, where N is the number of sequences, L the length, and k is 5 or 21"
    N, L, k = X.shape
    p = X.mean(axis=0)                 # frequency per symbol
    p = np.clip(p, eps, 1.0)           # avoid log(0)
    H = -(p * np.log(p)).sum(axis=-1)
    if neff:
        return np.exp(H.sum())
    elif pos:
        return np.exp(H)
    elif leff:
        return H.sum()/np.log(k)
    else:
        return np.exp(H.mean())


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


def get_center_data(X, w):
    mu = w @ X                        # (M,)
    Zx = X - mu                       # center
    Zw = np.sqrt(w)[:, None] * Zx     # sqrt(W) Zc
    return Zw


def effective_length(
    Z, w=None, tol=1e-12, signal_frac=None,
    zs=True, neff=False, reg=0., svd=True
):
    """
    Estimate the effective dimensional length of a dataset via the entropy of
    its variance spectrum.

    Parameters
    ----------
    Z : array_like, shape (N, L, k)
        Input tensor of N samples with L positions and k features per position.
    w : array_like, optional
        Sample weights of length N. If None, uniform weights are used.
    tol : float, optional
        Absolute numerical eigenvalue threshold. Components below this value
        are discarded before any signal-fraction truncation.
    signal_frac : float or None, optional
        Fraction of total retained variance to preserve. If in (0, 1], the
        spectrum is sorted in decreasing order and truncated to the smallest
        number of components whose cumulative variance is at least
        `signal_frac`. For example, `signal_frac=0.99` retains components
        explaining 99% of the variance.
    zs : bool, optional
        If True, apply zero-sum mean-removal constraint across features.
    neff : bool, optional
        If True, return a support estimate of the sequence space spanned by
        the MSA.
    reg : float, optional
        Diagonal regularization added to covariance in eigenvalue mode.
    svd : bool, optional
        If True, or when N < 5M, use SVD of centered data; otherwise use
        weighted covariance eigenvalues.

    Returns
    -------
    float
        Entropy-based effective rank of the feature space, optionally scaled
        to an effective alphabet-size support when `neff=True`.

    Notes
    -----
    The method computes the Shannon entropy of normalized singular values or
    covariance eigenvalues and converts it to an effective dimension called
    `L_eff`.
    """
    N, L, k = Z.shape
    Z = Z.reshape(N, -1)

    if zs:
        Z = to_zero_sum(Z, L, k)

    N, M = Z.shape

    if w is None:
        w = np.ones(N) / N

    if svd or N < 5 * M:
        Zw = get_center_data(Z, w)
        vals = np.linalg.svd(Zw, full_matrices=False, compute_uv=False)
        vals = (vals ** 2) / Zw.shape[0]
    else:
        C, mu, Zt = weighted_covariance(Z, w)
        vals = np.linalg.eigvalsh((C + C.T) / 2.0 + reg * np.eye(M))

    vals = vals[vals > tol]

    if vals.size == 0:
        return 0.0

    vals = np.sort(vals)[::-1]

    if signal_frac is not None:
        if not (0.0 < signal_frac <= 1.0):
            raise ValueError("signal_frac must be in (0, 1].")

        frac = np.cumsum(vals) / np.sum(vals)
        n_keep = np.searchsorted(frac, signal_frac) + 1
        vals = vals[:n_keep]

    p = vals / vals.sum()
    H = -np.sum(p * np.log(p))

    denom = k - 1 if zs else k
    Leff = np.exp(H) / denom

    if neff:
        return k ** Leff
    return float(Leff)


def spectral_entropy(C, tol=1e-12):
    lam_full, U_full = np.linalg.eigh(C)  # M eigenvalues, M×M eigenvectors
    mask = lam_full > tol
    lam = lam_full[mask]                  # r
    U   = U_full[:, mask]                 # M×r

    S = lam.sum()
    p = lam / S
    H = -np.sum(p * np.log(p))
    return H, lam, U


def cross_effective_length(X, Y, wx=None, wy=None, tol=1e-12, zs=True, alpha=1.):
    """
    Cross isotropy between two *independent* sample sets.
    X: (Nx, Lx, k)
    Y: (Ny, Ly, k)
    """

    Nx, Lx, k = X.shape
    Ny, Ly, k = Y.shape

    X = to_zero_sum(X.reshape(Nx, -1), Lx, k) if zs else X
    Y = to_zero_sum(Y.reshape(Ny, -1), Ly, k) if zs else Y

    if wx is None: wx = np.ones(Nx) / Nx
    if wy is None: wy = np.ones(Ny) / Ny
    wx = wx / wx.sum()
    wy = wy / wy.sum()

    Xc, Yc = get_center_data(X, wx), get_center_data(Y, wy)
    # independent cross moment
    Cxy = Xc @ Yc.T

    # singular spectrum
    s = np.linalg.svd(Cxy, compute_uv=False)
    s = s[s > tol]
    if s.size == 0:
        return 0.0

    p = s / s.sum()

    if alpha == 1:
        H = -np.sum(p * np.log(p))
        eff_rank = np.exp(H)
    else:
        eff_rank = np.sum(p**alpha) ** (1.0 / (1.0 - alpha))

    return float(eff_rank / (k - 1))

    # return float(np.exp(H) / (k - 1))

def avg_min_dist(Y, X):
    if len(X.shape) == 3:
        Xf, Yf = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    else:
        Xf, Yf = X, Y

    XY = Xf @ Yf.T
    XX = Xf @ Xf.T
    np.fill_diagonal(XX, 0)
    return (XY.max(1)[XX.max(1)>0] / (XX.max(1)[XX.max(1)>0] + 1e-12)).mean()


def leff(msa: list, **kwargs) -> float:
    return effective_length(msa_to_oh(msa), **kwargs)


def cross_leff(msa_x: list, msa_y: list, **kwargs) -> float:
    return cross_effective_length(msa_to_oh(msa_x), msa_to_oh(msa_y), **kwargs)

