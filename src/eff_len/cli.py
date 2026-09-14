"""Command line interface for eff_len."""

import argparse
import math
import sys

from .utils import read_msa, msa_to_oh
from .eff_len import effective_length, cross_effective_length

# LDDT > 0.7 thresholds from Opuu, PLoS Comput Biol
THRESHOLDS = {"AlphaFold": 4.96, "ESMFold": 18.69,
              "OmegaFold": 26.78, "RoseTTAFold": 37.27}


def _load(path, stype, fmt):
    msa = read_msa(path, seq_type=stype, fmt=fmt)
    return msa_to_oh(msa, seq_type=stype)


def _row(path, oh):
    N, L, k = oh.shape
    leff = effective_length(oh)
    return {"file": path, "N": N, "L": L, "L_eff": leff,
            "L_eff/L": leff / L if L else float("nan"),
            "log10_omega": leff * math.log10(k), "oh": oh}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="eff_len",
        description="Effective sequence length (L_eff) of an alignment.")
    ap.add_argument("msa", nargs="+",
                    help="alignment file(s): fasta, a2m/a3m or Stockholm")
    ap.add_argument("--stype", choices=["nuc", "prot"], default="prot",
                    help="sequence type (default: prot)")
    ap.add_argument("--fmt", choices=["auto", "fasta", "a3m"], default="auto",
                    help="force insert handling; 'a3m' strips lowercase "
                         "inserts, 'fasta' keeps them (default: auto)")
    ap.add_argument("--cross", metavar="MSA",
                    help="also report cross effective length against this "
                         "alignment (requires identical length)")
    ap.add_argument("--thresholds", action="store_true",
                    help="flag whether L_eff clears the published "
                         "structure-prediction thresholds")
    ap.add_argument("--tsv", action="store_true",
                    help="tab-separated output, one row per file")
    args = ap.parse_args(argv)

    if args.thresholds and args.stype != "prot":
        print("eff_len: --thresholds are protein structure-prediction "
              "cutoffs, ignored for --stype nuc", file=sys.stderr)

    rows = []
    for path in args.msa:
        try:
            rows.append(_row(path, _load(path, args.stype, args.fmt)))
        except (OSError, ValueError) as e:
            print(f"eff_len: {e}", file=sys.stderr)

    if not rows:
        return 1

    if args.tsv:
        cols = ["file", "N", "L", "L_eff", "L_eff/L", "log10_omega"]
        print("\t".join(cols))
        for r in rows:
            print("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float)
                            else str(r[c]) for c in cols))
    else:
        for r in rows:
            print(f"{r['file']}\n"
                  f"  sequences   N       = {r['N']}")
            print(f"  length      L       = {r['L']}")
            print(f"  effective   L_eff   = {r['L_eff']:.2f}")
            print(f"  normalised  L_eff/L = {r['L_eff/L']:.3f}")
            print(f"  support     log10 W = {r['log10_omega']:.2f}")
            if args.thresholds and args.stype == "prot":
                for model, tau in sorted(THRESHOLDS.items(),
                                         key=lambda x: x[1]):
                    ok = "pass" if r["L_eff"] > tau else "FAIL"
                    print(f"    {model:<12} tau={tau:>6.2f}  {ok}")

    if args.cross:
        try:
            ref = _load(args.cross, args.stype, args.fmt)
            ref_leff = effective_length(ref)
        except (OSError, ValueError) as e:
            print(f"eff_len: {e}", file=sys.stderr)
            return 1
        for r in rows:
            try:
                cross = cross_effective_length(r["oh"], ref)
            except Exception as e:
                print(f"eff_len: cross {r['file']} vs {args.cross}: {e}",
                      file=sys.stderr)
                continue
            denom = math.sqrt(r["L_eff"] * ref_leff)
            norm = cross / denom if denom else float("nan")
            print(f"cross {r['file']} vs {args.cross}: "
                  f"L_eff_cross = {cross:.2f}, normalised = {norm:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
