"""
JanusProWrapper: unified interface for Janus-Pro-1B generation and GRPO training.
Mirrors LlamaGenWrapper's API so the same training loop can drive both models.
"""
import math
import sys
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_prompt_str(processor, prompt_text: str) -> str:
    conversation = [
        {"role": "<|User|>",      "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    return sft_format + processor.image_start_tag


def _decode_to_pil(mmgpt, generated_tokens, img_size=384, patch_size=16):
    """Decode (B, 576) int token tensor to list of PIL images."""
    B = generated_tokens.shape[0]
    patches = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[B, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = patches.to(torch.float32).cpu().detach().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return [PIL.Image.fromarray(dec[i]) for i in range(B)]


class JanusProWrapper:
    def __init__(
        self,
        model_path: str = "deepseek-ai/Janus-Pro-1B",
        lora_config: Optional[dict] = None,
        cfg_weight: float = 5.0,
        temperature: float = 1.0,
        image_token_num: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        logprob_reduction: str = "sum_sqrt_len",
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.lora_config = lora_config or {}
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.image_token_num = image_token_num
        self.img_size = img_size
        self.patch_size = patch_size
        self.logprob_reduction = logprob_reduction
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None
        self._optimizer = None
        self._step = 0

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from janus.models import MultiModalityCausalLM, VLChatProcessor

        print(f"[JanusWrapper] Loading {self.model_path} ...")
        processor = VLChatProcessor.from_pretrained(self.model_path)
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        language_config = config.language_config
        language_config._attn_implementation = "eager"
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            language_config=language_config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        model = model.to(torch.bfloat16).cuda().eval()

        if self.lora_config:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=self.lora_config.get("r", 16),
                lora_alpha=self.lora_config.get("alpha", 32),
                lora_dropout=self.lora_config.get("dropout", 0.0),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            )
            scope = self.lora_config.get("target_scope", "language_model")
            if scope == "language_model":
                model.language_model = get_peft_model(model.language_model, lora_cfg)
                n_train = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in model.language_model.parameters())
            else:
                model = get_peft_model(model, lora_cfg)
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in model.parameters())
            print(f"[JanusWrapper] LoRA injected. Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

        self._processor = processor
        self._model = model
        self._optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )
        print("[JanusWrapper] Ready.")

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load()
        return self._processor

    # ------------------------------------------------------------------
    # Prompt batching helpers
    # ------------------------------------------------------------------

    def _tokenize_and_pad(self, prompts: List[str]):
        """
        Returns (tokens, attn_mask, uncond_tokens, uncond_mask) for CFG batch.
        All tensors: (B, max_len), left-padded.
        Unconditional prompt uses empty string, matching official Janus-Pro repo.
        """
        pad_id = self.processor.pad_id
        all_ids = [
            self.processor.tokenizer.encode(_build_prompt_str(self.processor, p))
            for p in prompts
        ]
        # unconditional: empty prompt string
        uncond_ids_single = self.processor.tokenizer.encode(
            _build_prompt_str(self.processor, "")
        )

        all_uncond_ids = [uncond_ids_single for _ in prompts]
        max_len = max(
            max(len(ids) for ids in all_ids),
            len(uncond_ids_single),
        )
        B = len(prompts)

        tokens = torch.full((B, max_len), pad_id, dtype=torch.int).cuda()
        attn_mask = torch.zeros(B, max_len, dtype=torch.long).cuda()
        uncond_tokens = torch.full((B, max_len), pad_id, dtype=torch.int).cuda()
        uncond_mask = torch.zeros(B, max_len, dtype=torch.long).cuda()

        for b, (ids, u_ids) in enumerate(zip(all_ids, all_uncond_ids)):
            # conditional — right-align
            offset = max_len - len(ids)
            tokens[b, offset:] = torch.tensor(ids, dtype=torch.int)
            attn_mask[b, offset:] = 1
            # unconditional — right-align
            u_offset = max_len - len(u_ids)
            uncond_tokens[b, u_offset:] = torch.tensor(u_ids, dtype=torch.int)
            uncond_mask[b, u_offset:] = 1

        return tokens, attn_mask, uncond_tokens, uncond_mask, max_len, all_ids

    def _interleave(self, cond, uncond):
        """Interleave cond (B, ...) and uncond (B, ...) → (2B, ...) even=cond odd=uncond."""
        B = cond.shape[0]
        out = torch.zeros(B * 2, *cond.shape[1:], dtype=cond.dtype, device=cond.device)
        out[0::2] = cond
        out[1::2] = uncond
        return out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_images(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        return_tokens: bool = False,
        return_logprobs: bool = False,
    ) -> dict:
        """
        Generate one image per prompt.
        When len(prompts) == 1: single-prompt batched path (fast, matches official style).
        When len(prompts) > 1: sequential per-prompt to avoid left-padding KV cache issues.
        Seeds are respected per-prompt in sequential mode.
        """
        cfg_weight  = cfg_weight  if cfg_weight  is not None else self.cfg_weight
        temperature = temperature if temperature is not None else self.temperature
        model = self.model
        model.eval()
        B = len(prompts)

        if seeds is not None:
            assert len(seeds) == B, f"Expected {B} seeds, got {len(seeds)}"

        if B > 1:
            # sequential: one prompt at a time to avoid left-padding KV cache corruption
            all_images, all_tokens, all_lp = [], [], []
            for idx, prompt in enumerate(prompts):
                seed = seeds[idx] if seeds is not None else None
                out = self._generate_single(
                    prompt, seed=seed, cfg_weight=cfg_weight, temperature=temperature,
                    return_tokens=return_tokens, return_logprobs=return_logprobs,
                )
                all_images.append(out["images"][0])
                if return_tokens:
                    all_tokens.append(out["generated_tokens"])
                if return_logprobs:
                    all_lp.append(out["token_logprobs"])
            result = {"images": all_images}
            if return_tokens:
                result["generated_tokens"] = torch.cat(all_tokens, dim=0)
            if return_logprobs:
                result["token_logprobs"] = torch.cat(all_lp, dim=0)
            return result

        # single-prompt fast path
        seed = seeds[0] if seeds is not None else None
        return self._generate_single(
            prompts[0], seed=seed, cfg_weight=cfg_weight, temperature=temperature,
            return_tokens=return_tokens, return_logprobs=return_logprobs,
        )

    @torch.no_grad()
    def _generate_single(
        self,
        prompt: str,
        seed: Optional[int] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        return_tokens: bool = False,
        return_logprobs: bool = False,
    ) -> dict:
        """Generate one image for one prompt. No batching, no padding ambiguity."""
        cfg_weight  = cfg_weight  if cfg_weight  is not None else self.cfg_weight
        temperature = temperature if temperature is not None else self.temperature
        model = self.model

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        tokens, attn_mask, uncond_tokens, uncond_mask, max_len, _ = self._tokenize_and_pad([prompt])

        # CFG interleaved batch: (2, max_len)
        cfg_tokens = self._interleave(tokens, uncond_tokens)
        cfg_mask   = self._interleave(attn_mask, uncond_mask)

        inputs_embeds   = model.language_model.get_input_embeddings()(cfg_tokens)
        generated_tokens = torch.zeros((1, self.image_token_num), dtype=torch.int).cuda()
        all_logprobs     = [] if return_logprobs else None

        pkv              = None
        past_attn_mask   = cfg_mask   # grows by 1 each step

        for i in range(self.image_token_num):
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=past_attn_mask,
                use_cache=True,
                past_key_values=pkv,
            )
            pkv    = outputs.past_key_values
            hidden = outputs.last_hidden_state
            logits = model.gen_head(hidden[:, -1, :])   # (2, vocab)

            logit_cond   = logits[0:1, :]
            logit_uncond = logits[1:2, :]
            logits_cfg   = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            probs      = torch.softmax(logits_cfg / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
            generated_tokens[:, i] = next_token.squeeze(-1)

            if return_logprobs:
                lp = torch.log_softmax(logits_cfg / temperature, dim=-1)
                all_logprobs.append(lp.gather(1, next_token))

            next_paired  = next_token.expand(2, 1).reshape(-1)     # (2,)
            img_embeds   = model.prepare_gen_img_embeds(next_paired)
            inputs_embeds = img_embeds.unsqueeze(1)                 # (2, 1, D)

            # extend attention mask by 1 for the newly generated token
            past_attn_mask = torch.cat([
                past_attn_mask,
                torch.ones(past_attn_mask.size(0), 1,
                           dtype=past_attn_mask.dtype,
                           device=past_attn_mask.device),
            ], dim=1)

        images = _decode_to_pil(model, generated_tokens, self.img_size, self.patch_size)

        result = {"images": images}
        if return_tokens:
            result["generated_tokens"] = generated_tokens.cpu()
        if return_logprobs:
            result["token_logprobs"] = torch.cat(all_logprobs, dim=1).cpu()  # (B, 576)
        return result

    # ------------------------------------------------------------------
    # Log prob recompute (training forward pass — single forward, no generation loop)
    # ------------------------------------------------------------------

    def _recompute_logprobs(
        self,
        prompts: List[str],
        generated_tokens: torch.Tensor,   # (B, 576) int, on cuda
        reduction: Optional[str] = None,
        return_per_token: bool = False,
    ) -> torch.Tensor:
        """
        Full-sequence teacher-forcing forward pass to get log probs.
        Uses conditional-only (no CFG) to keep memory manageable and match LlamaGen convention.
        Returns (B,) reduced log probs, or (B, 576) per-token log probs.
        """
        model = self.model
        red = reduction or self.logprob_reduction
        B = len(prompts)
        seq_len = self.image_token_num  # 576

        tokens, attn_mask, _, _, max_len, _ = self._tokenize_and_pad(prompts)

        # Prompt embeddings: (B, max_len, D)
        prompt_embeds = model.language_model.get_input_embeddings()(tokens)

        # Image token embeddings: (B, 576, D) via prepare_gen_img_embeds
        flat_img_tokens = generated_tokens.view(-1).long()               # (B*576,)
        flat_img_embeds = model.prepare_gen_img_embeds(flat_img_tokens)  # (B*576, D)
        img_embeds = flat_img_embeds.view(B, seq_len, -1)                # (B, 576, D)

        # Input: [prompt (max_len), img_embeds[:, :-1, :] (575)] — total max_len+575
        full_input = torch.cat([prompt_embeds, img_embeds[:, :-1, :]], dim=1)
        full_mask = torch.cat([
            attn_mask,
            torch.ones(B, seq_len - 1, dtype=torch.long, device=attn_mask.device)
        ], dim=1)

        outputs = model.language_model.model(
            inputs_embeds=full_input,
            attention_mask=full_mask,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state  # (B, max_len+575, D)
        all_logits = model.gen_head(hidden)  # (B, max_len+575, vocab)

        # Logits at positions [max_len-1 .. max_len+574] predict image tokens [0..575]
        img_logits = all_logits[:, max_len - 1: max_len + seq_len - 1, :]  # (B, 576, vocab)

        log_probs = F.log_softmax(img_logits / self.temperature, dim=-1)
        token_lp = log_probs.gather(-1, generated_tokens.long().unsqueeze(-1)).squeeze(-1)  # (B, 576)
        token_lp = token_lp.clamp(min=-20.0)

        if return_per_token:
            return token_lp

        if red == "sum":
            return token_lp.sum(dim=-1)
        elif red == "sum_sqrt_len":
            return token_lp.sum(dim=-1) / math.sqrt(seq_len)
        else:
            return token_lp.mean(dim=-1)

    def _recompute_logprobs_ref(self, prompts, generated_tokens, reduction=None):
        """Log probs under reference (base) model: zero LoRA B, compute, restore."""
        saved = {}
        for name, mod in self.model.named_modules():
            if hasattr(mod, "lora_B"):
                saved[name] = mod.lora_B.default.weight.data.clone()
                mod.lora_B.default.weight.data.zero_()
        with torch.no_grad():
            lp = self._recompute_logprobs(prompts, generated_tokens, reduction=reduction)
        for name, mod in self.model.named_modules():
            if name in saved:
                mod.lora_B.default.weight.data.copy_(saved[name])
        return lp

    # ------------------------------------------------------------------
    # GRPO training step
    # ------------------------------------------------------------------

    def train_grpo_step(
        self,
        batch,                          # List[BucketItem]
        reward_model,
        num_samples: int = 4,
        beta: float = 0.05,
        reward_mode: str = "pseudo_soft_grpo_target_heavy",
        advantage_eps: float = 1e-4,
        token_weighting: Optional[str] = None,   # None | "gcpo_lite"
        gcpo_config: Optional[dict] = None,
    ) -> dict:
        model = self.model
        prompts = [item.text for item in batch]
        B = len(batch)

        # 1. Generate num_samples images per prompt (no grad, collect tokens + logprobs)
        model.eval()
        all_tokens_list = []   # num_samples × (B, 576)
        all_pil_imgs = []      # flat: [b0_s0, b1_s0, ..., b0_s1, ...]

        for s in range(num_samples):
            out = self.generate_images(prompts, return_tokens=True, return_logprobs=False)
            all_tokens_list.append(out["generated_tokens"].cuda())  # (B, 576)
            all_pil_imgs.extend(out["images"])                       # B PIL images

        # 2. Score all images
        rewards = torch.zeros(B, num_samples)
        self._last_sample_details = []
        for s in range(num_samples):
            for i, item in enumerate(batch):
                pil_img = all_pil_imgs[s * B + i]
                soft_result = reward_model.score_image(pil_img, item, mode=reward_mode)
                hard_result = reward_model.score_image(pil_img, item, mode="hard_target")
                rewards[i, s] = soft_result["score"]
                self._last_sample_details.append({
                    "prompt_id": item.id,
                    "prompt": item.text,
                    "bucket": item.bucket,
                    "sample": s,
                    "soft_reward": float(soft_result["score"]),
                    "hard_reward": float(hard_result["score"]),
                    "question_scores": soft_result.get("question_scores", []),
                    "component_scores": soft_result.get("component_scores", {}),
                    "hard_component_scores": hard_result.get("component_scores", {}),
                })

        # 3. Group-relative advantages
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r  = rewards.std(dim=1, keepdim=True) + advantage_eps
        advantages = ((rewards - mean_r) / std_r).nan_to_num(nan=0.0)  # (B, G)

        # 4. Stack tokens: (B*G, 576)
        stacked_tokens = torch.stack(all_tokens_list, dim=1).reshape(B * num_samples, -1).cuda()
        rep_prompts = [p for p in prompts for _ in range(num_samples)]  # B*G prompts
        flat_advantages = advantages.reshape(-1).to(self.device)

        # 5. KL reference log probs
        ref_log_probs = None
        ref_logprob_mean = float("nan")
        if beta > 0.0:
            with torch.no_grad():
                ref_log_probs = self._recompute_logprobs_ref(rep_prompts, stacked_tokens)
            ref_logprob_mean = float(ref_log_probs.mean().item())

        # 6. Policy gradient (chunked to save memory)
        self._optimizer.zero_grad()
        model.train()

        chunk_size = B
        total = B * num_samples
        total_pg_loss = 0.0
        total_kl = 0.0
        all_seq_lp = []

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            c_tok   = stacked_tokens[start:end]
            c_prom  = rep_prompts[start:end]
            c_adv   = flat_advantages[start:end]
            weight  = (end - start) / total

            if token_weighting == "gcpo_lite":
                gcpo_cfg = gcpo_config or {}
                token_lp = self._recompute_logprobs(c_prom, c_tok, return_per_token=True)  # (chunk, 576)
                # Need logits for gcpo weights — run another forward to get logits
                # (Acceptable for diagnostic experiment; optimize later if needed)
                with torch.no_grad():
                    logits_for_weights = self._get_img_logits(c_prom, c_tok)  # (chunk, 576, vocab)
                tw = build_gcpo_lite_weights(
                    logits_for_weights.detach(),
                    grid_size=gcpo_cfg.get("grid_size", 24),
                    initial_ratio=gcpo_cfg.get("initial_ratio", 0.10),
                    entropy_gradient_ratio=gcpo_cfg.get("entropy_gradient_ratio", 0.20),
                    background_weight=gcpo_cfg.get("background_weight", 0.2),
                )
                tw = tw / tw.mean(dim=-1, keepdim=True).clamp_min(1e-6)
                lp = (tw * token_lp).sum(dim=-1) / tw.sum(dim=-1).clamp_min(1e-6)
            else:
                lp = self._recompute_logprobs(c_prom, c_tok)  # (chunk,)

            all_seq_lp.extend(lp.detach().cpu().tolist())
            pg = -(c_adv * lp).mean() * weight

            if beta > 0.0 and ref_log_probs is not None:
                kl_chunk = (lp - ref_log_probs[start:end]).mean() * weight
                chunk_loss = pg + beta * kl_chunk
                total_kl += kl_chunk.item()
            else:
                chunk_loss = pg

            chunk_loss.backward()
            total_pg_loss += pg.item()

            # sanitize NaN grads after each chunk
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # LoRA diagnostics
        lora_params = [p for n, p in model.named_parameters()
                       if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
        lora_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in lora_params if p.grad is not None) ** 0.5
            if lora_params else 0.0
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            self.max_grad_norm,
        ).item()
        self._optimizer.step()
        lora_weight_norm = (
            sum(p.data.float().norm().item() ** 2 for p in lora_params) ** 0.5
            if lora_params else 0.0
        )

        reward_stds = rewards.std(dim=1).cpu()
        flat_adv_abs = advantages.abs().reshape(-1).cpu()
        seq_lp_t = torch.tensor(all_seq_lp)
        self._step += 1

        return {
            "loss": total_pg_loss + beta * total_kl,
            "pg_loss": total_pg_loss,
            "kl_loss": total_kl,
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_min": rewards.min().item(),
            "reward_max": rewards.max().item(),
            "lr": self._optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "lora_weight_norm": lora_weight_norm,
            "lora_grad_norm": lora_grad_norm,
            "percent_groups_zero_std": float((reward_stds < 1e-6).float().mean().item() * 100),
            "mean_group_reward_std": float(reward_stds.mean().item()),
            "mean_abs_advantage": float(flat_adv_abs.mean().item()),
            "fraction_nonzero_advantage": float((flat_adv_abs > 1e-6).float().mean().item()),
            "seq_logprob_mean": float(seq_lp_t.mean().item()),
            "seq_logprob_std": float(seq_lp_t.std().item()) if len(seq_lp_t) > 1 else 0.0,
            "ref_logprob_mean": ref_logprob_mean,
            "logprob_reduction": self.logprob_reduction,
            "reward_mode": reward_mode,
            "token_weighting": token_weighting or "none",
        }

    def _get_img_logits(self, prompts, generated_tokens):
        """Return (B, 576, vocab) image logits (no grad) for gcpo weight computation."""
        model = self.model
        B = len(prompts)
        seq_len = self.image_token_num
        tokens, attn_mask, _, _, max_len, _ = self._tokenize_and_pad(prompts)
        prompt_embeds = model.language_model.get_input_embeddings()(tokens)
        flat_img_embeds = model.prepare_gen_img_embeds(generated_tokens.view(-1).long())
        img_embeds = flat_img_embeds.view(B, seq_len, -1)
        full_input = torch.cat([prompt_embeds, img_embeds[:, :-1, :]], dim=1)
        full_mask = torch.cat([
            attn_mask,
            torch.ones(B, seq_len - 1, dtype=torch.long, device=attn_mask.device)
        ], dim=1)
        with torch.no_grad():
            outputs = model.language_model.model(inputs_embeds=full_input, attention_mask=full_mask, use_cache=False)
            img_logits = model.gen_head(outputs.last_hidden_state[:, max_len - 1: max_len + seq_len - 1, :])
        return img_logits

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        lora_state = {k: v for k, v in self.model.state_dict().items()
                      if "lora_A" in k or "lora_B" in k}
        torch.save({
            "lora_state": lora_state,
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "step": self._step,
            "model_path": self.model_path,
            "lora_config": self.lora_config,
        }, path)
        print(f"[JanusWrapper] Saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["lora_state"], strict=False)
        if self._optimizer and ckpt.get("optimizer"):
            self._optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
        print(f"[JanusWrapper] Loaded checkpoint from {path} (step={self._step})")

    def current_checkpoint_id(self) -> str:
        return f"step_{self._step:06d}"


# ------------------------------------------------------------------
# GCPO-lite token weighting
# ------------------------------------------------------------------

def build_gcpo_lite_weights(
    logits: torch.Tensor,           # (B, 576, vocab)
    grid_size: int = 24,
    initial_ratio: float = 0.10,
    entropy_gradient_ratio: float = 0.20,
    background_weight: float = 0.2,
) -> torch.Tensor:
    """
    Build per-token importance weights for GCPO-lite.
    High weight for: early (top-left) tokens and tokens at high entropy-gradient positions.
    Returns (B, 576) float tensor.
    """
    B, n_tokens, _ = logits.shape  # n_tokens = 576

    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # (B, 576)

    # entropy gradient: absolute change between adjacent positions
    entropy_grad = torch.zeros_like(entropy)
    entropy_grad[:, 1:] = (entropy[:, 1:] - entropy[:, :-1]).abs()

    n_initial = max(1, int(n_tokens * initial_ratio))      # e.g. 57
    n_grad    = max(1, int(n_tokens * entropy_gradient_ratio))  # e.g. 115

    weights = torch.full((B, n_tokens), background_weight, device=logits.device, dtype=torch.float32)
    weights[:, :n_initial] = 1.0

    _, top_idx = entropy_grad.topk(n_grad, dim=-1)         # (B, n_grad)
    weights.scatter_(1, top_idx, 1.0)

    return weights
