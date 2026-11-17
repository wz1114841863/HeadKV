"""CPU-only matched Pre/Post Head budget exploration on the existing twelve inputs."""
import argparse
import os


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','codec-root','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]=str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from experiments.head_analysis.analysis.pre_rope_head_budget import run_analysis
    run_analysis(args.bundle,args.codec_root,args.output_dir)


if __name__=='__main__':
    main()
