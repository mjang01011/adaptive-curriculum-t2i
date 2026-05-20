"""
Entry point: python -m adaptive_curriculum.train.run_experiment --config ... --strategy ucb
"""
import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive curriculum experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment.yaml")
    parser.add_argument("--strategy", type=str, choices=["uniform", "static", "ucb"], default="ucb")
    parser.add_argument("--output-root", type=str, default=None, help="Override config output_root")
    parser.add_argument("--data-root", type=str, default=None, help="Override config data_root")
    parser.add_argument("--pretrained-root", type=str, default=None, help="Override config pretrained_root")
    parser.add_argument("--no-model", action="store_true", help="Dry run without loading LlamaGen (heuristic only)")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_curriculum_steps")
    return parser.parse_args()


def main():
    args = parse_args()

    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)

    # apply CLI overrides
    if args.output_root:
        config.paths.output_root = args.output_root
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.pretrained_root:
        config.paths.pretrained_root = args.pretrained_root
    if args.num_steps:
        config.training.num_curriculum_steps = args.num_steps

    config._use_real_model = not args.no_model

    # ensure data root has toy data if empty
    data_root = config.paths.data_root
    buckets_root = Path(data_root) / "buckets"
    if not buckets_root.exists() or not any(buckets_root.iterdir()):
        print(f"[run_experiment] No bucket data found at {buckets_root}. Generating toy data...")
        from adaptive_curriculum.data.build_buckets import build_toy_data
        build_toy_data(str(data_root), n_train=20, n_val=10)

    from adaptive_curriculum.train.train_supervised_curriculum import run_curriculum_training
    run_dir = run_curriculum_training(config, strategy=args.strategy)
    print(f"[run_experiment] Finished. Results at: {run_dir}")


if __name__ == "__main__":
    main()
