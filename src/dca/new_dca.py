"""Code derived from aDAM
"""

import argparse
import torch
import itertools
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .utils import read_fasta, msa_to_oh, get_one_hot
from sklearn.metrics import r2_score


def get_freq_single_point(data, weights=None):
    if weights is not None:
        return (data * weights[:, None, None]).sum(dim=0)
    else:
        return data.mean(dim=0)


def get_freq_two_points(data, weights=None):
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    if weights is not None:
        we_data_oh = data_oh * weights[:, None]
    else:
        we_data_oh = data_oh * 1./M
    fij = we_data_oh.T @ data_oh  # Compute weighted sum
    return fij.reshape(L, q, L, q)


def set_zerosum_gauge(coupling):
    coupling -= coupling.mean(dim=1, keepdim=True) + \
                coupling.mean(dim=3, keepdim=True) - \
                coupling.mean(dim=(1, 3), keepdim=True)
    return coupling


def compute_energy_confs(x, h_parms, j_parms):
    M, L, q = x.shape

    # Flatten along the last two dimensions (L*q) for batch processing
    x_oh = x.reshape(M, L * q)
    bias_oh = h_parms.view(-1)  # Flatten bias
    couplings_oh = j_parms.reshape(L * q, L * q)

    # Compute energy contributions
    field = - torch.matmul(x_oh, bias_oh)  # Shape (M,)
    couplings = - 0.5 * torch.einsum('mi,ij,mj->m', x_oh, couplings_oh, x_oh)  # Shape (M,)

    return field + couplings


def init_chains(q, num_chains, L):
    chains = torch.randint(low=0, high=q, size=(num_chains, L))
    return get_one_hot(chains)


def gibbs(chains, h_parms, j_parms, beta, nb_steps, verbose=False):
    """Performs a Gibbs sweep over the chains."""
    N, L, q = chains.shape
    for steps in range(nb_steps):
        residue_idxs = torch.randperm(L)
        for i in residue_idxs:
            couplings_residue = j_parms[i].reshape(q, L * q)
            logit_residue = h_parms[i].unsqueeze(0) + chains.reshape(N, L * q) @ couplings_residue.T
            chains[:, i, :] = get_one_hot(torch.multinomial(torch.softmax(beta * logit_residue, dim=-1), 1), num_classes=q).squeeze(1)
        if verbose:
            print(steps)
    return chains


def fit_model(exp_conf, max_step, min_pearson, lr=0.01, N=10000, nb_gibbs=10, beta=1, l1=1e-4):
    M, L, q = exp_conf.shape
    h_parms = torch.zeros((L, q))
    j_parms = torch.zeros((L, q, L, q))

    fi, fij = get_freq_single_point(exp_conf), get_freq_two_points(exp_conf)
    chains = init_chains(q, N, L)
    pi, pij = get_freq_single_point(chains), get_freq_two_points(chains)

    halt_condition = lambda s, p: s >= max_step or p >= min_pearson

    # pearson = pearsonr(pij.flatten(), fij.flatten())[0]
    # pearson = 1-(((fij.flatten()-pij.flatten())**2).sum() / ((fij.flatten()-fij.flatten().mean())**2).sum()).item()
    pearson = r2_score(fij.flatten(), pij.flatten())
    step = 0

    while not halt_condition(step, pearson):
        grad_i = (pi - fi)
        grad_ij = (pij - fij)
        h_parms -= lr * grad_i + l1 * torch.sign(h_parms)
        j_parms -= lr * grad_ij + l1 * torch.sign(j_parms)
        j_parms = set_zerosum_gauge(j_parms)
        chains = gibbs(chains, h_parms, j_parms, beta, nb_gibbs)
        pi, pij = get_freq_single_point(chains), get_freq_two_points(chains)
        # pearson = 1 - ((pij.flatten()-fij.flatten())**2).sum() / ((fij.flatten()-fij.flatten().mean())**2).sum().item()
        pearson = r2_score(fij.flatten(), pij.flatten())
        # pearson = pearsonr(fij.flatten(), pij.flatten())
        step += 1
        # print(step, pearson)

    return h_parms, j_parms


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input')
    parser.add_argument('-ms', '--max_steps', type=int, default=200)
    parser.add_argument('-mp', '--min_pearson', type=float, default=0.90)
    parser.add_argument('-lr', '--lr', type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_arguments()
    exp_seq = read_fasta(args.input)
    msa_oh = msa_to_oh(exp_seq)

    h_parms, j_parms = fit_model(torch.tensor(msa_oh), args.max_steps, args.min_pearson, lr=args.lr)


if __name__ == '__main__':
    main()
