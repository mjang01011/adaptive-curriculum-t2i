"""
GRPOTrainer for LlamaGen, adapted from AR-GRPO (modified_grpo_trainer.py).

Loss:
    per_token_loss = -min(r_t * A,  clip(r_t, 1-ε, 1+ε) * A)
    r_t = exp(logp_θ(t) - logp_old(t))   per-token probability ratio

KL (reverse, AR-GRPO style):
    per_token_kl = exp(logp_ref - logp_θ) - (logp_ref - logp_θ) - 1

Reduction (GRPO-style):
    loss = mean_over_batch[ sum_over_tokens(loss_t) / num_tokens ]

Advantages:
    A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)   per prompt group
    Groups with std < 0.03 are zeroed out (noise guard).
    scale_rewards=False → subtract mean only, don't divide by std.

num_iterations > 1:
    rollout() computes old_per_token_logps once before any gradient steps.
    _update_iteration() is called num_iterations times on the same fixed rollout,
    so r_t drifts away from 1 across iterations and PPO clipping becomes active.
"""
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOW_STD_THRESHOLD = 0.03   # groups below this are zeroed


class GRPOTrainer:
    def __init__(
        self,
        wrapper,
        num_generations: int = 8,
        num_iterations: int = 3,
        beta: float = 1.0,
        epsilon: float = 0.2,
        scale_rewards: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.wrapper = wrapper
        self.G = num_generations
        self.num_iterations = num_iterations
        self.beta = beta
        self.epsilon = epsilon
        self.scale_rewards = scale_rewards
        self.max_grad_norm = max_grad_norm
        self._step = 0

    # ------------------------------------------------------------------
    # Per-token log probs (no reduction)
    # ------------------------------------------------------------------

    def _compute_per_token_logps(
        self,
        image_tokens: torch.Tensor,   # (B, seq_len) int64
        c_indices: torch.Tensor,       # (B, 120, 2048) float32
        c_emb_masks: torch.Tensor,     # (B, 120)
    ) -> torch.Tensor:
        """
        Returns (B, seq_len) per-token log probs under current policy.
        Dropout disabled; uncond_prob zeroed.
        Caller must ensure gpt is in float32 before calling.
        """
        gpt = self.wrapper.gpt
        gpt.train()
        saved_uncond_prob = gpt.cls_embedding.uncond_prob
        gpt.cls_embedding.uncond_prob = 0.0
        dropout_mods = [m for m in gpt.modules() if isinstance(m, nn.Dropout)]
        for m in dropout_mods:
            m.eval()

        try:
            tokens_long = image_tokens.long()
            c_cast = c_indices.float()

            logits, _ = gpt(
                idx=tokens_long[:, :-1],
                cond_idx=c_cast,
                input_pos=None,
                targets=None,
                mask=None,
                valid=None,
            )
            logits = logits.float().nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
            log_p = F.log_softmax(logits, dim=-1)
            token_lp = log_p.gather(-1, tokens_long.unsqueeze(-1)).squeeze(-1)  # (B, seq_len)
            return token_lp.clamp(min=-20.0)

        finally:
            gpt.cls_embedding.uncond_prob = saved_uncond_prob
            for m in dropout_mods:
                m.train()

    def _compute_per_token_logps_ref(
        self,
        image_tokens: torch.Tensor,
        c_indices: torch.Tensor,
        c_emb_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token log probs under reference model (LoRA-B zeroed = base weights)."""
        gpt = self.wrapper.gpt
        saved = {}
        for name, mod in gpt.named_modules():
            if hasattr(mod, "lora_B"):
                saved[name] = mod.lora_B.weight.data.clone()
                mod.lora_B.weight.data.zero_()
        with torch.no_grad():
            lp = self._compute_per_token_logps(image_tokens, c_indices, c_emb_masks)
        for name, mod in gpt.named_modules():
            if name in saved:
                mod.lora_B.weight.data.copy_(saved[name])
        return lp

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        batch: list,
        reward_model,
        reward_mode: str = "pseudo_soft_grpo_target_heavy",
        t5_cache=None,
    ) -> dict:
        """
        1. Generate G token sequences per prompt.
        2. Decode + score all B*G images.
        3. Compute per-group advantages (with low-variance group zeroing).
        4. Compute old_per_token_logps (no-grad, current policy).
        5. Compute ref_per_token_logps (no-grad, LoRA-B zeroed).

        Timing is tracked at zero overhead using perf_counter calls and
        returned in the dict for W&B logging.
        """
        from autoregressive.models.generate import generate
        import torchvision.transforms.functional as TF

        t_rollout_start = time.perf_counter()

        wrapper = self.wrapper
        B = len(batch)
        G = self.G
        device = wrapper.device

        # ── text conditioning ─────────────────────────────────────────
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning(batch, t5_cache)

        # ── generate G sequences per prompt ───────────────────────────
        wrapper.gpt.eval()
        all_tokens: List[torch.Tensor] = []
        all_pil_imgs: list = []
        qzshape = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]

        t_gen_start = time.perf_counter()
        with torch.no_grad():
            for s in range(G):
                index_sample = generate(
                    wrapper.gpt, c_indices, wrapper.latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=wrapper.cfg_scale_train,
                    temperature=wrapper.temperature,
                    top_k=wrapper.top_k,
                    top_p=wrapper.top_p,
                    sample_logits=True,
                )  # (B, seq_len)
                all_tokens.append(index_sample)

                decoded = wrapper.vq_model.decode_code(index_sample, qzshape)
                for i in range(B):
                    img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
                    all_pil_imgs.append(TF.to_pil_image(img_t.cpu()))

        wrapper._disable_kv_cache()
        t_gen = time.perf_counter() - t_gen_start

        # ── score all B*G images ──────────────────────────────────────
        t_score_start = time.perf_counter()
        rewards = torch.zeros(B, G)
        sample_details = []
        for s in range(G):
            for i, item in enumerate(batch):
                pil_img = all_pil_imgs[s * B + i]
                result = reward_model.score_image(pil_img, item, mode=reward_mode)
                score = float(result["score"])
                rewards[i, s] = score
                sample_details.append({
                    "prompt_id":       item.id,
                    "prompt":          item.text,
                    "bucket":          item.bucket,
                    "sample_idx":      s,
                    "reward":          score,
                    "question_scores": result.get("question_scores", []),
                    "component_scores": result.get("component_scores", {}),
                    "reward_debug":    result.get("reward_debug", {}),
                })
        t_score = time.perf_counter() - t_score_start

        # ── per-group advantage normalization ─────────────────────────
        mean_r = rewards.mean(dim=1, keepdim=True)   # (B, 1)
        std_r  = rewards.std(dim=1, keepdim=True)    # (B, 1)
        advantages = rewards - mean_r
        if self.scale_rewards:
            advantages = advantages / (std_r + 1e-4)
        advantages = advantages.nan_to_num(nan=0.0)  # (B, G)

        # Low-variance group protection: zero out groups where reward
        # differences are too small to be meaningful signal (likely noise).
        low_std_mask = std_r < _LOW_STD_THRESHOLD   # (B, 1)
        advantages = torch.where(low_std_mask.expand_as(advantages),
                                 torch.zeros_like(advantages), advantages)
        pct_zeroed = float(low_std_mask.float().mean().item() * 100)

        # ── flatten tokens / conditioning ─────────────────────────────
        stacked_tokens = torch.stack(all_tokens, dim=1).reshape(B * G, -1).to(device)
        rep_c     = c_indices.repeat_interleave(G, dim=0)
        rep_masks = c_emb_masks.repeat_interleave(G, dim=0)
        flat_advantages = advantages.reshape(-1).to(device)  # (B*G,)

        # ── cast to float32 once for all logp computations ───────────
        wrapper.gpt.float()

        # ── old_per_token_logps (no-grad, current policy) ────────────
        t_logp_start = time.perf_counter()
        with torch.no_grad():
            old_per_token_logps = self._compute_per_token_logps(
                stacked_tokens, rep_c, rep_masks
            ).detach()

        # ── ref_per_token_logps (no-grad, LoRA-B zeroed) ─────────────
        ref_per_token_logps = None
        if self.beta > 0.0:
            with torch.no_grad():
                ref_per_token_logps = self._compute_per_token_logps_ref(
                    stacked_tokens, rep_c, rep_masks
                ).detach()
        t_logp = time.perf_counter() - t_logp_start

        t_rollout = time.perf_counter() - t_rollout_start

        return {
            "stacked_tokens":              stacked_tokens,
            "rep_c":                       rep_c,
            "rep_masks":                   rep_masks,
            "flat_advantages":             flat_advantages,
            "old_per_token_logps":         old_per_token_logps,
            "ref_per_token_logps":         ref_per_token_logps,
            "rewards":                     rewards,
            "sample_details":              sample_details,
            "percent_groups_zeroed_low_std": pct_zeroed,
            # timing (seconds)
            "t_gen":     t_gen,
            "t_score":   t_score,
            "t_logp":    t_logp,
            "t_rollout": t_rollout,
        }

    # ------------------------------------------------------------------
    # Single update iteration on a fixed rollout
    # ------------------------------------------------------------------

    def _update_iteration(self, rollout: dict) -> dict:
        """One gradient update using a fixed (frozen) rollout."""
        wrapper = self.wrapper
        opt = wrapper._optimizer

        stacked_tokens      = rollout["stacked_tokens"]
        rep_c               = rollout["rep_c"]
        rep_masks           = rollout["rep_masks"]
        flat_advantages     = rollout["flat_advantages"]
        old_per_token_logps = rollout["old_per_token_logps"]
        ref_per_token_logps = rollout["ref_per_token_logps"]

        t_update_start = time.perf_counter()

        opt.zero_grad()
        per_token_logps = self._compute_per_token_logps(stacked_tokens, rep_c, rep_masks)

        # PPO clipped loss
        r_t      = torch.exp(per_token_logps - old_per_token_logps)
        r_t_clip = torch.clamp(r_t, 1 - self.epsilon, 1 + self.epsilon)
        adv = flat_advantages.unsqueeze(1)
        per_token_loss = -torch.min(r_t * adv, r_t_clip * adv)

        # Reverse KL
        per_token_kl = None
        if self.beta > 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # GRPO reduction
        seq_len = per_token_logps.shape[1]
        loss = (per_token_loss.sum(-1) / seq_len).mean()

        loss.backward()

        for p in wrapper.gpt.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        lora_params = [p for n, p in wrapper.gpt.named_parameters()
                       if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
        lora_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in lora_params if p.grad is not None) ** 0.5
            if lora_params else 0.0
        )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in wrapper.gpt.parameters() if p.requires_grad],
            self.max_grad_norm,
        ).item()

        opt.step()

        t_update = time.perf_counter() - t_update_start

        with torch.no_grad():
            mean_kl   = float(per_token_kl.mean().item()) if per_token_kl is not None else float("nan")
            mean_r_t  = float(r_t.mean().item())
            clip_frac = float((r_t != r_t_clip).float().mean().item())
            logp_mean = float(per_token_logps.detach().mean().item())

        return {
            "loss":           loss.item(),
            "mean_kl":        mean_kl,
            "mean_r_t":       mean_r_t,
            "clip_frac":      clip_frac,
            "grad_norm":      grad_norm,
            "lora_grad_norm": lora_grad_norm,
            "seq_logp_mean":  logp_mean,
            "t_update":       t_update,
        }

    # ------------------------------------------------------------------
    # Public train step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: list,
        reward_model,
        reward_mode: str = "pseudo_soft_grpo_target_heavy",
        t5_cache=None,
    ) -> dict:
        """
        Full GRPO step: rollout → num_iterations updates → cast back to bf16.
        Returns metrics from the last iteration plus reward/timing stats.
        """
        t_step_start = time.perf_counter()

        rollout = self.rollout(batch, reward_model, reward_mode=reward_mode, t5_cache=t5_cache)
        rewards = rollout["rewards"]

        iter_metrics = {}
        t_update_total = 0.0
        for _ in range(self.num_iterations):
            iter_metrics = self._update_iteration(rollout)
            t_update_total += iter_metrics.pop("t_update")
            self._step += 1

        self.wrapper.gpt.to(dtype=self.wrapper.dtype)

        t_step = time.perf_counter() - t_step_start

        reward_stds  = rewards.std(dim=1).cpu()
        flat_adv_abs = rollout["flat_advantages"].detach().abs().cpu()

        return {
            **iter_metrics,
            "mean_reward":                   rewards.mean().item(),
            "reward_std":                    rewards.std().item(),
            "reward_min":                    rewards.min().item(),
            "reward_max":                    rewards.max().item(),
            "mean_abs_advantage":            float(flat_adv_abs.mean().item()),
            "percent_groups_zero_std":       float((reward_stds < 1e-6).float().mean().item() * 100),
            "percent_groups_zeroed_low_std": rollout["percent_groups_zeroed_low_std"],
            "mean_group_reward_std":         float(reward_stds.mean().item()),
            "lr":                            self.wrapper._optimizer.param_groups[0]["lr"],
            "step":                          self._step,
            # timing
            "t_gen_s":          rollout["t_gen"],
            "t_score_s":        rollout["t_score"],
            "t_logp_s":         rollout["t_logp"],
            "t_rollout_s":      rollout["t_rollout"],
            "t_update_total_s": t_update_total,
            "t_step_s":         t_step,
            "sample_details":   rollout["sample_details"],
        }

    # ------------------------------------------------------------------
    # Checkpoint delegation
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        self.wrapper.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        self.wrapper.load_checkpoint(path)
        self._step = self.wrapper._step
