# eff_len

Spectral measure of diversity for multiple sequence alignments.

## Overview

This repo contains the code to reproduce the results in *"A spectral framework for measuring diversity in multiple sequence alignments"*. It introduces a measure, `L_eff`, that estimates the diversity (or amount of information) contained in a multiple sequence alignment. `L_eff` allows a faithful comparison between MSAs, as well as between generated datasets.

## Install

Requirements: NumPy.

```bash
pip install eff_len
```

From source:

```bash
git clone https://github.com/vaiteaopuu/effective_length
cd effective_length
pip install .
```

## Usage

### Python

```python
from eff_len import read_fasta, msa_to_oh, effective_length

msa = read_fasta("data/RF00028.fa", seq_type="nuc")

msa_oh = msa_to_oh(msa, seq_type="nuc")
N, L, k = msa_oh.shape
L_eff = effective_length(msa_oh)

print(N, L, L_eff, L_eff / L)
# 2611 251 35.88477058938092 0.14296721350350963
```

`cross_effective_length` and `leff` are also exported, for comparing two alignments and for the convenience wrapper respectively.

## Repository content

| Path | Description |
| --- | --- |
| `src/eff_len/` | The package itself |
| `analysis/` | Notebooks and scripts |
| `data/` | Example alignments |
| `reproducibility.org` | Code snippets reproducing the figures in the paper |

## Data sources

The data used in these analyses were extracted from:

- C. Lambert *et al.* (2025) *Nat. Commun.*
- F. Calvanese *et al.* (2024) *NAR*
- M. Mirdita *et al.* (2027) *NAR*

## Citation

If you use this code, please cite:

```bibtex
@article{opuu_spectral,
  title   = {A spectral framework for measuring diversity in multiple sequence alignments},
  author  = {Opuu, Vaitea},
  year    = {2026}
}
```

## License

MIT
