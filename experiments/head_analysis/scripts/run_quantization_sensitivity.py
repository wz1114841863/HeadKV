"""Run the frozen V3-B Calib or Validation cohort; model weights must be local."""

from experiments.head_analysis.runners.quantization_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
