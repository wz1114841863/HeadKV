"""Run V2 D1/D2 Attention and KV association analysis from local numerical packages."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.attention_kv_joint import run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--b2-bundle", type=Path, required=True, help="completed B2 bundle retaining the B1 RCO copy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)
    run_analysis(args.bundle, args.reference_bundle, args.output_dir, b2_bundle=args.b2_bundle,
                 bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
