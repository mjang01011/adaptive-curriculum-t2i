"""
Entry point: python -m adaptive_curriculum.train.run_experiment --config ... --strategy ucb
"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive curriculum experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment.yaml")
    parser.add_argument("--strategy", type=str, choices=["uniform", "static", "ucb"], default="ucb")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--pretrained-root", type=str, default=None)
    parser.add_argument("--t5-cache-dir", type=str, default=None)
    parser.add_argument("--no-model", action="store_true", help="Dry run without LlamaGen (heuristic reward only)")
    parser.add_argument("--num-steps", type=int, default=None)
    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    return parser.parse_args()


def main():
    args = parse_args()

    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)

    # CLI overrides
    if args.output_root:
        config.paths.output_root = args.output_root
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.pretrained_root:
        config.paths.pretrained_root = args.pretrained_root
    if args.t5_cache_dir:
        config.paths.t5_cache_dir = args.t5_cache_dir
    if args.num_steps:
        config.training.num_curriculum_steps = args.num_steps
    if args.wandb:
        config.logging.use_wandb = True
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.logging.wandb_entity = args.wandb_entity
    if args.run_name:
        config.logging.run_name = args.run_name

    config._use_real_model = not args.no_model

    # verify data exists
    data_root = Path(config.paths.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    from adaptive_curriculum.train.train_supervised_curriculum import run_curriculum_training
    run_dir = run_curriculum_training(config, strategy=args.strategy)
    print(f"[run_experiment] Finished. Results at: {run_dir}")


if __name__ == "__main__":
    main()
