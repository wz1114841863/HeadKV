"""Create a new offline dual-scope Page RCO report for one completed B1 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.head_analysis.analysis.page_rco import PageRCOError, analyze_completed_run, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True, help="new JSON report; never overwrites")
    parser.add_argument("--k", type=int, action="append", help="repeatable; defaults to 8,16,32")
    parser.add_argument("--delta", type=int, action="append", help="repeatable; defaults to 1,2,4,...,256")
    args = parser.parse_args()
    try:
        write_json_new(
            Path(args.output),
            analyze_completed_run(
                Path(args.run_dir),
                k_values=(8, 16, 32) if args.k is None else args.k,
                deltas=(1, 2, 4, 8, 16, 32, 64, 128, 256) if args.delta is None else args.delta,
            ),
        )
    except (PageRCOError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
