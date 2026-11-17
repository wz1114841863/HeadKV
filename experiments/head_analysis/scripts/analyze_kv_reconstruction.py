"""V3-A: prepare and freeze inputs, validate all choices, then open Test once."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.head_analysis.analysis.kv_reconstruction import prepare, run_validation, run_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "validation", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", action="append", type=Path)
    parser.add_argument("--fixed-inputs", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    if args.phase == "prepare":
        if not args.run_dir or not args.fixed_inputs:
            parser.error("prepare requires --run-dir and --fixed-inputs")
        prepare(args.run_dir, args.fixed_inputs, args.output_dir)
    elif args.phase == "validation":
        run_validation(args.output_dir)
    else:
        run_test(args.output_dir)
    print(f"V3 {args.phase} completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
