import re
from RNA import fold_compound
import torch

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


# scoring function (you already have this)
def eval_structure(seq, structure):
    return fold_compound(seq).eval_structure(structure)

def encode_msa(seqs, alphabet, gap="-"):
    stoi = {c:i for i,c in enumerate(alphabet)}
    L = len(seqs[0])
    def encode_seq(s):
        s = s[:L].ljust(L, gap)
        return [stoi.get(c, stoi[gap]) for c in s]

    X = torch.tensor([encode_seq(s) for s in seqs], dtype=torch.long)
    return X


def sample_prior(model, n, sigma=1.0):
    # z ~ N(0, sigma^2 I)
    d = model.mu.out_features
    dev = next(model.parameters()).device
    z = torch.randn(n, d, device=dev) * sigma
    return model.decode(z)  # [n,L,S]
