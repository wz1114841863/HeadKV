"""Compare fixed Head profiles with input-dependent variation in a V2 numeric bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.head_identity import run_analysis, run_reference_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="completed V2 representation numeric bundle")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing results are preserved")
    parser.add_argument("--reference-bundle", type=Path, help="fixed Pilot numeric bundle for Question A")
    parser.add_argument("--fixed-inputs", type=Path, help="frozen prompts.jsonl providing target length and Pair IDs")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)
    if args.reference_bundle is not None:
        if args.fixed_inputs is None:
            parser.error("--reference-bundle requires --fixed-inputs")
        run_reference_analysis(args.bundle, args.reference_bundle, args.output_dir,
                               fixed_inputs=args.fixed_inputs,
                               bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    else:
        run_analysis(args.bundle, args.output_dir)
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
