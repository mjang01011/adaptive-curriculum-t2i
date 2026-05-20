"""
Helper script to launch all three baseline strategies on Modal in sequence.

Usage:
  python adaptive_curriculum/modal/launch_experiment.py --num-steps 200
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--strategies", nargs="+", default=["uniform", "static", "ucb"])
    parser.add_argument("--dry-run", action="store_true", help="Use --no-model flag (no real LlamaGen)")
    args = parser.parse_args()

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Launching strategy: {strategy}")
        print(f"{'='*60}")

        if args.dry_run:
            cmd = [
                "modal", "run",
                "adaptive_curriculum/modal/modal_app.py::dry_run_no_gpu_test",
                "--strategy", strategy,
                "--num-steps", str(args.num_steps),
            ]
        else:
            cmd = [
                "modal", "run",
                "adaptive_curriculum/modal/modal_app.py::run_experiment_modal",
                "--strategy", strategy,
                "--num-steps", str(args.num_steps),
            ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: strategy={strategy} failed with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

    print("\nAll strategies completed successfully.")


if __name__ == "__main__":
    main()
