"""
CARGOTrainer: Component-Aware Reward-Grounded token advantage GRPO.

Extends GRPOTrainer with per-token advantages derived from CARGO importance masks.

Token advantage formula:
    A_cargo[flat_idx, t] = lambda_base * A_scalar[flat_idx]
                         + (1 - lambda_base) * sum_c( w_c * A_c[b, g] * mask_c[b, t] )

where:
    A_scalar  = standard group-normalized GRPO advantage (scalar per sample)
    A_c[b, g] = per-component advantage (component c, batch item b, sample g),
                normalized independently within each (batch item, component) group
    mask_c[b] = CARGO winner-aligned 16×16 smoothed importance mask for component c
    w_c       = 1 / n_active_components (uniform)
    lambda_base = base scalar weight (default 0.25)

Stability gates:
    Per-component variance gate: if std_c[b] < 0.05, that component contributes
    nothing to the residual for batch item b (signal too weak to trust).

    Final token advantages are clamped to [-2, 2].
"""
import os
import sys
import time

import torch

# Ensure project root is importable when this module is run directly or
# when CARGO/ is not on sys.path yet.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GRPO.trainer  import GRPOTrainer
from CARGO.masks   import compute_cargo_mask, compute_cargo_mask_pixel
from CARGO.rewards import META_KEYS

_COMP_VAR_GATE = 0.05   # per-component group std below this → skip (variance gate)
_ADV_CLAMP     = 2.0    # clip token advantages to [-ADV_CLAMP, ADV_CLAMP]


class CARGOTrainer(GRPOTrainer):
    """
    Subclass of GRPOTrainer that replaces scalar per-sample advantages with
    per-token advantages weighted by CARGO component importance masks.
    """

    def __init__(
        self,
        wrapper,
        num_generations: int  = 8,
        num_iterations:  int  = 1,           # CARGO default is 1 (more stable than 3)
        beta:            float = 1.0,
        epsilon:         float = 0.2,
        scale_rewards:   bool  = True,
        max_grad_norm:   float = 1.0,
        logprob_mode:    str   = "cfg",
        cargo_lambda_base:  float = 0.25,    # weight of scalar GRPO base term
        cargo_mask_floor:   float = 0.30,    # minimum token importance (soft floor)
        cargo_mask_source:  str   = "pixel", # "pixel" (default) or "vq" (ablation)
        elite_sft_alpha:    float = 0.0,     # coefficient for elite SFT auxiliary loss (0 = off)
        elite_sft_frac:     float = 0.25,    # fraction of top samples per group used for SFT
    ):
        super().__init__(
            wrapper=wrapper,
            num_generations=num_generations,
            num_iterations=num_iterations,
            beta=beta,
            epsilon=epsilon,
            scale_rewards=scale_rewards,
            max_grad_norm=max_grad_norm,
            logprob_mode=logprob_mode,
        )
        self.cargo_lambda_base = cargo_lambda_base
        self.cargo_mask_floor  = cargo_mask_floor
        self.cargo_mask_source = cargo_mask_source
        self.elite_sft_alpha   = elite_sft_alpha
        self.elite_sft_frac    = elite_sft_frac

    @property
    def latent_size(self) -> int:
        return self.wrapper.latent_size

    # ------------------------------------------------------------------
    # Image decoding for pixel masks
    # ------------------------------------------------------------------

    def _decode_rollout_images(
        self,
        stacked_tokens: torch.Tensor,   # (B*G, seq_len)
        B: int,
        G: int,
    ) -> list:
        """
        Decode stacked_tokens back to (B*G,) PIL images, ordered [b*G+g].
        Called only when cargo_mask_source="pixel". VQ decode is fast (~ms each).
        """
        import torchvision.transforms.functional as TF
        wrapper   = self.wrapper
        ls        = wrapper.latent_size
        qzshape   = [1, wrapper.codebook_embed_dim, ls, ls]
        pil_images = []
        with torch.no_grad():
            for idx in range(B * G):
                decoded = wrapper.vq_model.decode_code(
                    stacked_tokens[idx : idx + 1], qzshape
                )
                img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
                pil_images.append(TF.to_pil_image(img_t.cpu()))
        return pil_images

    # ------------------------------------------------------------------
    # CARGO advantage computation
    # ------------------------------------------------------------------

    def _extract_component_rewards(
        self, sample_details: list, B: int, G: int
    ) -> dict:
        """
        Build {comp_key: (B, G) float32 tensor} from sample_details.

        sample_details ordering: index [s*B + i] = generation sample s, batch item i.
        Output tensor ordering:  [b, g] = batch item b, generation sample g.
        """
        all_keys: set = set()
        for d in sample_details:
            all_keys.update((d.get("component_scores") or {}).keys())

        comp_rewards = {}
        for key in all_keys:
            mat = torch.zeros(B, G, dtype=torch.float32)
            for g in range(G):
                for b in range(B):
                    scores = sample_details[g * B + b].get("component_scores") or {}
                    mat[b, g] = float(scores.get(key, 0.5))
            comp_rewards[key] = mat
        return comp_rewards

    def _compute_cargo_advantages(
        self,
        flat_advantages:  torch.Tensor,   # (B*G,) scalar GRPO advantages, on device
        comp_rewards:     dict,           # {key: (B, G) cpu float32}
        stacked_tokens:   torch.Tensor,   # (B*G, seq_len) int64, on device
        B:                int,
        G:                int,
        pil_images:       list = None,    # (B*G,) PIL images for pixel mode
    ) -> torch.Tensor:
        """
        Return per-token CARGO advantages of shape (B*G, seq_len).

        When cargo_mask_source="pixel" (default), masks are computed from decoded
        RGB patch L1 distances — semantically consistent across generations.
        When cargo_mask_source="vq", falls back to VQ token-identity differences
        (ablation; produces near-uniform masks for LlamaGen due to codebook variance).

        stacked_tokens layout: [b*G + g, :] = batch item b, generation g.
        pil_images layout:     pil_images[b*G + g] = PIL for batch item b, gen g.
        comp_rewards layout:   comp_rewards[key][b, g].
        """
        device  = stacked_tokens.device
        seq_len = stacked_tokens.shape[1]

        tokens_bg = stacked_tokens.reshape(B, G, seq_len)   # (B, G, seq_len)
        use_pixel = (self.cargo_mask_source == "pixel") and (pil_images is not None)

        cargo_keys = [k for k in comp_rewards if k not in META_KEYS]
        if not cargo_keys:
            return flat_advantages.unsqueeze(1).expand(-1, seq_len).clone()

        residual_sum    = torch.zeros(B, G, seq_len, dtype=torch.float32)   # cpu
        surviving_count = torch.zeros(B, dtype=torch.float32)               # cpu

        for key in cargo_keys:
            R_c = comp_rewards[key]                          # (B, G) cpu

            mean_c = R_c.mean(dim=1, keepdim=True)
            std_c  = R_c.std(dim=1, keepdim=True)
            A_c    = (R_c - mean_c) / (std_c + 1e-4)
            A_c    = A_c.nan_to_num(nan=0.0)
            low_var = (std_c.squeeze(1) < _COMP_VAR_GATE)   # (B,) bool

            for b in range(B):
                if low_var[b]:
                    continue

                R_c_b = R_c[b].to(device)                   # (G,)

                if use_pixel:
                    images_b = [pil_images[b * G + g] for g in range(G)]
                    mask_b = compute_cargo_mask_pixel(
                        images_b, R_c_b,
                        latent_size=self.latent_size,
                        mask_floor=self.cargo_mask_floor,
                    )
                else:
                    mask_b = compute_cargo_mask(
                        tokens_bg[b], R_c_b,
                        latent_size=self.latent_size,
                        mask_floor=self.cargo_mask_floor,
                    )  # (seq_len,) on device

                A_c_b = A_c[b].to(device)    # (G,)
                residual_sum[b] += (A_c_b.unsqueeze(1) * mask_b.unsqueeze(0)).cpu()
                surviving_count[b] += 1.0

        norms    = surviving_count.clamp(min=1.0).reshape(B, 1, 1)
        residual = residual_sum / norms

        residual_flat   = residual.reshape(B * G, seq_len).to(device)
        adv_scalar_flat = flat_advantages.unsqueeze(1)

        cargo_adv = (
            self.cargo_lambda_base * adv_scalar_flat
          + (1.0 - self.cargo_lambda_base) * residual_flat
        )
        return cargo_adv.clamp(-_ADV_CLAMP, _ADV_CLAMP)

    # ------------------------------------------------------------------
    # Override _update_iteration to use per-token advantages
    # ------------------------------------------------------------------

    def _update_iteration(self, rollout: dict) -> dict:
        """
        One gradient update using CARGO per-token advantages when present,
        otherwise falls back to scalar GRPO advantages (safe drop-in).
        """
        import torch.nn as nn

        wrapper = self.wrapper
        opt     = wrapper._optimizer

        stacked_tokens      = rollout["stacked_tokens"]
        rep_c               = rollout["rep_c"]
        rep_masks           = rollout["rep_masks"]
        flat_advantages     = rollout["flat_advantages"]
        old_per_token_logps = rollout["old_per_token_logps"]
        ref_per_token_logps = rollout["ref_per_token_logps"]
        # Per-token advantages (B*G, seq_len) if CARGO, else fall back to scalar
        cargo_adv = rollout.get("cargo_token_advantages", None)

        t_update_start = time.perf_counter()

        opt.zero_grad()
        per_token_logps = self._compute_per_token_logps(stacked_tokens, rep_c, rep_masks)

        r_t      = torch.exp(per_token_logps - old_per_token_logps)
        r_t_clip = torch.clamp(r_t, 1 - self.epsilon, 1 + self.epsilon)

        adv = cargo_adv if cargo_adv is not None else flat_advantages.unsqueeze(1)
        per_token_loss = -torch.min(r_t * adv, r_t_clip * adv)

        per_token_kl = None
        if self.beta > 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        seq_len = per_token_logps.shape[1]
        loss    = (per_token_loss.sum(-1) / seq_len).mean()

        # ── Elite-replay SFT auxiliary loss ───────────────────────────────
        # Take top elite_sft_frac samples per prompt group and maximize their
        # log probs. Reuses per_token_logps already computed above (no extra
        # forward pass). Helps anchor the model to compositionally correct
        # generations and counteracts policy drift / reward hacking.
        sft_loss_val = 0.0
        if self.elite_sft_alpha > 0.0:
            rewards_bg = rollout.get("rewards")  # (B, G)
            if rewards_bg is not None:
                B_r, G_r    = rewards_bg.shape
                n_elite     = max(1, int(G_r * self.elite_sft_frac))
                elite_idxs  = []
                for b in range(B_r):
                    top_g = rewards_bg[b].topk(n_elite).indices
                    for g in top_g.tolist():
                        elite_idxs.append(b * G_r + g)
                if elite_idxs:
                    et = torch.tensor(elite_idxs, device=per_token_logps.device)
                    sft_loss = -per_token_logps[et].mean()
                    loss     = loss + self.elite_sft_alpha * sft_loss
                    sft_loss_val = float(sft_loss.detach().item())

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
            "sft_loss":       sft_loss_val,
            "mean_kl":        mean_kl,
            "mean_r_t":       mean_r_t,
            "clip_frac":      clip_frac,
            "grad_norm":      grad_norm,
            "lora_grad_norm": lora_grad_norm,
            "seq_logp_mean":  logp_mean,
            "t_update":       t_update,
        }

    # ------------------------------------------------------------------
    # train_step: rollout → CARGO advantages → update(s)
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch:        list,
        reward_model,
        reward_mode:  str  = "grpo_attr_contrastive_rubric_v2",
        t5_cache      = None,
    ) -> dict:
        """
        Full CARGO step:
          1. rollout() — generate G sequences, score, compute base GRPO advantages
          2. _extract_component_rewards() — (B, G) tensor per component key
          3. _compute_cargo_advantages()  — (B*G, seq_len) per-token advantages
          4. num_iterations × _update_iteration()
        """
        t_step_start = time.perf_counter()

        rollout = self.rollout(batch, reward_model, reward_mode=reward_mode, t5_cache=t5_cache)
        rewards = rollout["rewards"]
        B = len(batch)
        G = self.G

        # Decode images for pixel-space masks (fast; reuses already-generated tokens)
        pil_images = None
        if self.cargo_mask_source == "pixel":
            pil_images = self._decode_rollout_images(rollout["stacked_tokens"], B, G)

        # Compute per-token CARGO advantages and inject into rollout dict
        comp_rewards = self._extract_component_rewards(rollout["sample_details"], B, G)
        cargo_token_advantages = self._compute_cargo_advantages(
            rollout["flat_advantages"],
            comp_rewards,
            rollout["stacked_tokens"],
            B, G,
            pil_images=pil_images,
        )
        rollout["cargo_token_advantages"] = cargo_token_advantages

        iter_metrics   = {}
        t_update_total = 0.0
        for _ in range(self.num_iterations):
            iter_metrics    = self._update_iteration(rollout)
            t_update_total += iter_metrics.pop("t_update")
            self._step     += 1

        self.wrapper.gpt.to(dtype=self.wrapper.dtype)
        t_step = time.perf_counter() - t_step_start

        reward_stds   = rewards.std(dim=1).cpu()
        flat_adv_abs  = rollout["flat_advantages"].detach().abs().cpu()
        cargo_adv_abs = cargo_token_advantages.detach().abs().cpu()

        return {
            **iter_metrics,
            "mean_reward":                   rewards.mean().item(),
            "reward_std":                    rewards.std().item(),
            "reward_min":                    rewards.min().item(),
            "reward_max":                    rewards.max().item(),
            "mean_abs_advantage":            float(flat_adv_abs.mean().item()),
            "mean_abs_cargo_advantage":      float(cargo_adv_abs.mean().item()),
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
            "cargo_comp_rewards": {k: v.tolist() for k, v in comp_rewards.items()},
        }
