"""Run the approved fixed-PCA-subspace diagnostic without model inference."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    from experiments.head_analysis.analysis.kv_subspace_diagnostic import run_diagnostic
    run_diagnostic(args.bundle, args.output_dir)
    print(f"Subspace diagnostic completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
