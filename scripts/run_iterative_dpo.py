"""
Iterative semi-offline DPO (optionally + SFT) for LlamaGen.

Each round:
  1. Generate G=2 pairs from current model
  2. Train DPO (+ optional SFT) for N epochs
  3. Pass best_checkpoint.pt to next round

Usage:
  python scripts/run_iterative_dpo.py \
    --train-jsonl $PROJ/data/attribute_binding/attribute_binding_train_500.jsonl \
    --val-jsonl   $PROJ/data/attribute_binding/attribute_binding_val_20.jsonl \
    --output-dir  $PROJ/outputs_dpo/iterative_dpo \
    --repo-root   $PROJ/LlamaGen \
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path     $PRETRAINED/t5-ckpt \
    --num-rounds 3 --num-prompts 256 \
    --epochs 3 2 2 --lr 1e-5 1e-5 5e-6 \
    --sft-lambda 0.0 \
    --wandb-project llamagen-dpo-iterative
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parents[1]


def run(cmd, label):
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")
    print("  " + " \\\n    ".join(cmd))
    print()
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"\n  [{label}] done in {elapsed/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train-jsonl",    required=True)
    parser.add_argument("--val-jsonl",      required=True)
    parser.add_argument("--num-prompts",    type=int, default=256)
    # model
    parser.add_argument("--repo-root",      required=True)
    parser.add_argument("--gpt-ckpt",       required=True)
    parser.add_argument("--vq-ckpt",        required=True)
    parser.add_argument("--t5-path",        required=True)
    parser.add_argument("--cfg-scale",      type=float, default=2.0)
    # lora
    parser.add_argument("--lora-r",         type=int, default=16)
    parser.add_argument("--lora-alpha",     type=int, default=32)
    # iterative config
    parser.add_argument("--num-rounds",     type=int, default=3)
    parser.add_argument("--epochs",         type=int, nargs="+", default=None,
                        help="Epochs per round. If fewer values than rounds, last value repeats.")
    parser.add_argument("--lr",             type=float, nargs="+", default=None,
                        help="LR per round. If fewer values than rounds, last value repeats.")
    parser.add_argument("--beta",           type=float, default=0.2)
    parser.add_argument("--sft-lambda",     type=float, default=0.0)
    parser.add_argument("--batch-size",     type=int, default=2)
    parser.add_argument("--base-seed",      type=int, default=0,
                        help="Seeds for round r = [base_seed+2r, base_seed+2r+1]")
    parser.add_argument("--reward-mode",    default="pseudo_soft_grpo_target_heavy")
    # eval
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--val-seeds",      type=int, nargs="+", default=[0, 1])
    parser.add_argument("--val-prompts",    type=int, default=20)
    # output
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--wandb-project",  default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pad per-round lists to num_rounds (repeat last value)
    def pad(lst, default):
        if lst is None:
            return [default] * args.num_rounds
        return lst + [lst[-1]] * max(0, args.num_rounds - len(lst))

    epochs_per_round = pad(args.epochs, 3)
    lr_per_round     = pad(args.lr, 1e-5)

    print(f"\n[iterative-dpo] {args.num_rounds} rounds  sft_lambda={args.sft_lambda}")
    print(f"  epochs/round : {epochs_per_round[:args.num_rounds]}")
    print(f"  lr/round     : {lr_per_round[:args.num_rounds]}")

    prev_ckpt = None   # None = use base model weights
    round_summary = []

    for r in range(args.num_rounds):
        seed_a = args.base_seed + 2 * r
        seed_b = args.base_seed + 2 * r + 1
        pair_dir = out_dir / f"pairs_round{r+1}"
        dpo_dir  = out_dir / f"train_round{r+1}"
        epochs   = epochs_per_round[r]
        lr       = lr_per_round[r]

        print(f"\n\n{'━'*70}")
        print(f"  ROUND {r+1}/{args.num_rounds}  seeds=[{seed_a},{seed_b}]  "
              f"epochs={epochs}  lr={lr}  prev_ckpt={prev_ckpt or 'base'}")
        print(f"{'━'*70}")

        # ── pair generation ───────────────────────────────────────────────────
        gen_cmd = [
            sys.executable,
            str(_REPO / "scripts_janus" / "generate_and_score_llamagen_pairs_g2.py"),
            "--input-jsonl",  args.train_jsonl,
            "--output-dir",   str(pair_dir),
            "--num-prompts",  str(args.num_prompts),
            "--seeds",        str(seed_a), str(seed_b),
            "--repo-root",    args.repo_root,
            "--gpt-ckpt",     args.gpt_ckpt,
            "--vq-ckpt",      args.vq_ckpt,
            "--t5-path",      args.t5_path,
            "--cfg-scale",    str(args.cfg_scale),
            "--reward-mode",  args.reward_mode,
            "--save-tokens",
        ]
        if prev_ckpt:
            gen_cmd += ["--init-checkpoint", str(prev_ckpt)]

        run(gen_cmd, f"Round {r+1} — pair generation")

        # ── DPO training ──────────────────────────────────────────────────────
        train_cmd = [
            sys.executable,
            str(_REPO / "scripts" / "train_llamagen_dpo_from_pairs.py"),
            "--pairs-jsonl",       str(pair_dir / "pairs.jsonl"),
            "--val-jsonl",         args.val_jsonl,
            "--output-dir",        str(dpo_dir),
            "--repo-root",         args.repo_root,
            "--gpt-ckpt",          args.gpt_ckpt,
            "--vq-ckpt",           args.vq_ckpt,
            "--t5-path",           args.t5_path,
            "--cfg-scale",         str(args.cfg_scale),
            "--lora-r",            str(args.lora_r),
            "--lora-alpha",        str(args.lora_alpha),
            "--lr",                str(lr),
            "--beta",              str(args.beta),
            "--sft-lambda",        str(args.sft_lambda),
            "--epochs",            str(epochs),
            "--batch-size",        str(args.batch_size),
            "--eval-every-steps",  str(args.eval_every_steps),
            "--val-seeds",         *[str(s) for s in args.val_seeds],
            "--val-prompts",       str(args.val_prompts),
        ]
        if prev_ckpt:
            train_cmd += ["--init-checkpoint", str(prev_ckpt)]
        if args.wandb_project:
            train_cmd += ["--wandb-project", args.wandb_project]

        run(train_cmd, f"Round {r+1} — DPO training")

        # ── read round summary ────────────────────────────────────────────────
        summary_path = dpo_dir / "summary.json"
        round_info = {"round": r + 1, "pair_dir": str(pair_dir), "dpo_dir": str(dpo_dir)}
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            round_info.update({
                "best_val_reward":  s.get("best_val_reward"),
                "baseline_reward":  s.get("baseline_reward"),
                "delta_reward":     s.get("delta_reward"),
            })
        round_summary.append(round_info)

        prev_ckpt = dpo_dir / "best_checkpoint.pt"
        print(f"\n  [round {r+1}] best_val_r={round_info.get('best_val_reward', '?')}  "
              f"delta={round_info.get('delta_reward', '?')}  "
              f"next init → {prev_ckpt}")

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  ITERATIVE DPO COMPLETE")
    print(f"{'═'*70}")
    for r in round_summary:
        print(f"  Round {r['round']}:  best_val_r={r.get('best_val_reward','?')}  "
              f"delta={r.get('delta_reward','?')}")

    with open(out_dir / "iterative_summary.json", "w") as f:
        json.dump(round_summary, f, indent=2)
    print(f"\n  summary → {out_dir / 'iterative_summary.json'}")
    print(f"  final checkpoint → {prev_ckpt}")


if __name__ == "__main__":
    main()
