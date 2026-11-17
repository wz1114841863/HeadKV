"""Run the frozen, CPU-only Dense KV storage model on six Calib/six Validation inputs."""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-set", "source-root", "bundle", "b1-bundle", "codec-root", "output-dir"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    from experiments.head_analysis.analysis.dense_kv_memory import run_analysis
    run_analysis(args.run_set, args.source_root, args.bundle, args.b1_bundle, args.codec_root, args.output_dir)


if __name__ == "__main__":
    main()
