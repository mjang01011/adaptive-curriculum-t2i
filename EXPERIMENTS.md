# Experiment Log

## How to run

```bash
python -m adaptive_curriculum.train.run_experiment \
    --config adaptive_curriculum/configs/experiment.yaml \
    --experiment adaptive_curriculum/configs/experiments/<name>.yaml \
    --strategy ucb \
    [other path args]
```

The `--experiment` file is merged on top of the base config — only specify keys that differ.

---

## Experiments

### exp-001 — baseline UCB (current run on viscam)
- **Date**: 2026-05-20
- **Config**: base `experiment.yaml` (no experiment override)
- **Strategy**: UCB
- **Key settings**: 200 steps, 32 grad steps/bucket, num_samples=8, lr=1e-5, batch=4
- **W&B run**: llamagen-ucb-curriculum / 20260520_013019_ucb
- **Status**: Running (killed at ~step 9 or timed out at 48h)
- **Findings**:
  - No clear reward improvement after 9 steps
  - grad_norm ~0.03 (very small — lr likely too low)
  - complex_composition degraded 0.71 → 0.55 (LoRA interference / eval noise)
  - attribute_binding only bucket with slightly positive improvement_ma
  - UCB correctly identifying attribute_binding as most promising
  - ~54 min/step → would take ~7.5 days for full 200 steps (too slow)

### exp-002 — pilot_fast (planned)
- **Date**: —
- **Config**: `experiments/pilot_fast.yaml`
- **Strategy**: UCB
- **Key changes from exp-001**: 50 steps, 8 grad steps/bucket, num_samples=4, lr=5e-5, eval 8 prompts×4 samples
- **Expected runtime**: ~8h on viscam
- **Goal**: Verify reward actually improves with higher LR and faster UCB reselection

### exp-003 — ablation uniform (planned)
- **Date**: —
- **Config**: `experiments/ablation_uniform.yaml`
- **Strategy**: uniform
- **Goal**: Baseline to compare UCB bucket selection benefit vs random
- **Note**: Run same settings as exp-002 for fair comparison

### exp-004 — full_ucb (planned, after exp-002 confirms learning)
- **Date**: —
- **Config**: `experiments/full_ucb.yaml`
- **Strategy**: UCB
- **Key changes**: 200 steps, 16 grad steps/bucket, lr=3e-5, beta=0.01 KL, 16 prompts×2 samples eval
- **Goal**: Full production run once pilot confirms the reward signal is clean
