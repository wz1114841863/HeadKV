"""Compare fixed Head profiles with input-dependent variation in a V2 numeric bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.head_identity import run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="completed V2 representation numeric bundle")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing results are preserved")
    args = parser.parse_args(argv)
    run_analysis(args.bundle, args.output_dir)
    print(f"results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
