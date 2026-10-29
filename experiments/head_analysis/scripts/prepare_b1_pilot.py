"""Check the frozen B1 pilot environment and write a non-overwriting launch plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.head_analysis.runners.b1_runner import B1RunnerError
from experiments.head_analysis.runners.pilot_preflight import (
    EXPECTED_DATA_SHA256,
    PilotPreflightError,
    build_pilot_plan,
    write_json_new,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True, help="exclusive CUDA device, for example cuda:4")
    parser.add_argument("--run-dir", required=True, help="new B1 artifact directory")
    parser.add_argument("--baseline-output", required=True, help="new baseline JSON path")
    parser.add_argument("--output", required=True, help="new immutable preflight-plan JSON path")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--python-executable", default=".venv/bin/python")
    parser.add_argument("--expected-data-sha256", default=EXPECTED_DATA_SHA256)
    parser.add_argument("--minimum-free-memory-gib", type=float, default=32.0)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=200.0)
    args = parser.parse_args()
    if args.minimum_free_memory_gib <= 0 or args.minimum_free_disk_gib <= 0:
        raise SystemExit("minimum free memory/disk values must be positive")
    try:
        plan = build_pilot_plan(
            input_path=Path(args.input), manifest_path=Path(args.manifest), tokenizer_path=Path(args.tokenizer),
            model_path=Path(args.model), device=args.device, run_dir=Path(args.run_dir),
            baseline_output=Path(args.baseline_output), expected_commit=args.expected_commit,
            python_executable=args.python_executable,
            minimum_free_memory_bytes=int(args.minimum_free_memory_gib * 1024**3),
            minimum_free_disk_bytes=int(args.minimum_free_disk_gib * 1024**3),
            expected_data_sha256=args.expected_data_sha256,
        )
        write_json_new(Path(args.output), plan)
    except (PilotPreflightError, B1RunnerError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
