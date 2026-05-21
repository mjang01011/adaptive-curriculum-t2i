"""
LlamaGenWrapper: unified interface for sampling and supervised fine-tuning.
Hides LlamaGen repo internals from the curriculum training loop.
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


class LlamaGenWrapper:
    def __init__(
        self,
        repo_root: str,
        vq_ckpt: str,
        gpt_ckpt: str,
        gpt_model: str = "GPT-XL",
        image_size: int = 256,
        downsample_size: int = 16,
        codebook_size: int = 16384,
        codebook_embed_dim: int = 8,
        cls_token_num: int = 120,
        t5_path: str = "pretrained_models/t5-ckpt",
        t5_model_type: str = "flan-t5-xl",
        t5_feature_max_len: int = 120,
        cfg_scale: float = 2.0,
        cfg_scale_train: Optional[float] = None,
        temperature: float = 1.0,
        top_k: int = 1000,
        top_p: float = 1.0,
        precision: str = "bf16",
        device: Optional[str] = None,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
        logprob_reduction: str = "sum_sqrt_len",
    ):
        self.repo_root = repo_root
        self.vq_ckpt = vq_ckpt
        self.gpt_ckpt = gpt_ckpt
        self.gpt_model_name = gpt_model
        self.image_size = image_size
        self.downsample_size = downsample_size
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.cls_token_num = cls_token_num
        self.t5_path = t5_path
        self.t5_model_type = t5_model_type
        self.t5_feature_max_len = t5_feature_max_len
        self.cfg_scale = cfg_scale
        self.cfg_scale_train = cfg_scale_train if cfg_scale_train is not None else cfg_scale
        self.logprob_reduction = logprob_reduction
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.precision = precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self._dtype_map = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        self.dtype = self._dtype_map.get(precision, torch.bfloat16)
        self.latent_size = image_size // downsample_size

        # lazily loaded components
        self._vq_model = None
        self._gpt_model = None
        self._t5_model = None
        self._optimizer = None
        self._scaler = GradScaler("cuda") if precision == "fp16" else None
        self._step = 0

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_vq(self):
        from tokenizer.tokenizer_image.vq_model import VQ_models
        m = VQ_models["VQ-16"](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim,
        )
        m.to(self.device)
        m.eval()
        ckpt = torch.load(self.vq_ckpt, map_location="cpu")
        m.load_state_dict(ckpt["model"])
        del ckpt
        for p in m.parameters():
            p.requires_grad = False
        return m

    def _load_gpt(self, ckpt_path: Optional[str] = None):
        from autoregressive.models.gpt import GPT_models
        m = GPT_models[self.gpt_model_name](
            block_size=self.latent_size ** 2,
            cls_token_num=self.cls_token_num,
            model_type="t2i",
        ).to(device=self.device, dtype=self.dtype)

        path = ckpt_path or self.gpt_ckpt
        ckpt = torch.load(path, map_location="cpu")
        key = next((k for k in ("model", "module", "state_dict") if k in ckpt), None)
        weight = ckpt[key] if key else ckpt
        m.load_state_dict(weight, strict=False)
        del ckpt

        if self.use_lora:
            from adaptive_curriculum.model.lora_utils import inject_lora, freeze_base_model, count_trainable_parameters
            target = self.lora_config.get("target_modules", ["wqkv", "wo"])
            rank = self.lora_config.get("rank", 8)
            alpha = self.lora_config.get("alpha", 16.0)
            dropout = self.lora_config.get("dropout", 0.05)
            start_layer = self.lora_config.get("start_layer", 0)
            inject_lora(m, target_modules=target, rank=rank, alpha=alpha, dropout=dropout, start_layer=start_layer)
            freeze_base_model(m)
            n_trainable = count_trainable_parameters(m)
            print(f"[LlamaGenWrapper] LoRA injected. Trainable params: {n_trainable:,}")
        else:
            for p in m.parameters():
                p.requires_grad = True

        return m

    def _load_t5(self):
        from language.t5 import T5Embedder
        return T5Embedder(
            device=self.device,
            local_cache=True,
            cache_dir=self.t5_path,
            dir_or_name=self.t5_model_type,
            torch_dtype=self.dtype,
            model_max_length=self.t5_feature_max_len,
        )

    @property
    def vq_model(self):
        if self._vq_model is None:
            self._vq_model = self._load_vq()
        return self._vq_model

    @property
    def gpt(self):
        if self._gpt_model is None:
            self._gpt_model = self._load_gpt()
            self._optimizer = torch.optim.AdamW(
                [p for p in self._gpt_model.parameters() if p.requires_grad],
                lr=self.learning_rate,
            )
        return self._gpt_model

    @property
    def t5(self):
        if self._t5_model is None:
            self._t5_model = self._load_t5()
        return self._t5_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_images(
        self,
        prompts: List[str],
        out_dir: str,
        prompt_ids: Optional[List[str]] = None,
        bucket_names: Optional[List[str]] = None,
        num_samples_per_prompt: int = 1,
        seed: Optional[int] = None,
        cached_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[str]:
        """
        Generate images for each prompt.

        cached_embeddings: optional dict {item_id -> tensor(1, seq_len, 2048)} produced by
            extract_t5_embeddings.py. When provided, T5 is skipped entirely — the cache
            tensors are padded and used directly. This eliminates T5 inference from the
            evaluation hot path.
        """
        from torchvision.utils import save_image
        from autoregressive.models.generate import generate

        if seed is not None:
            torch.manual_seed(seed)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        prompt_ids = prompt_ids or [f"prompt_{i:06d}" for i in range(len(prompts))]
        bucket_names = bucket_names or ["unknown"] * len(prompts)

        self.gpt.eval()
        image_paths = []

        with torch.no_grad():
            if cached_embeddings is not None:
                # Build (B, max_len, 2048) + mask from cache — no T5 call
                c_indices, c_emb_masks = self._pack_cached_embeddings(
                    prompt_ids, cached_embeddings
                )
            else:
                caption_embs, emb_masks = self.t5.get_text_embeddings(prompts)
                new_caption_embs = []
                for emb, mask in zip(caption_embs, emb_masks):
                    valid = int(mask.sum().item())
                    new_caption_embs.append(torch.cat([emb[valid:], emb[:valid]]))
                new_caption_embs = torch.stack(new_caption_embs)
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                c_indices = new_caption_embs * new_emb_masks[:, :, None]
                c_emb_masks = new_emb_masks

            for k in range(num_samples_per_prompt):
                qzshape = [len(c_indices), self.codebook_embed_dim, self.latent_size, self.latent_size]
                index_sample = generate(
                    self.gpt, c_indices, self.latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=self.cfg_scale,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    sample_logits=True,
                )
                samples = self.vq_model.decode_code(index_sample, qzshape)

                for i, (pid, bucket) in enumerate(zip(prompt_ids, bucket_names)):
                    fname = f"{bucket}_{pid}_sample{k}.png"
                    img_path = str(out_path / fname)
                    save_image(samples[i:i+1], img_path, normalize=True, value_range=(-1, 1))
                    image_paths.append(img_path)

        return image_paths

    def _pack_cached_embeddings(
        self,
        prompt_ids: List[str],
        cache: Dict[str, torch.Tensor],
    ):
        """
        Convert {id: tensor(1, seq_len, 2048)} entries into left-padded
        (B, t5_feature_max_len, 2048) + mask, matching LlamaGen's sampling convention.
        """
        max_len = self.t5_feature_max_len
        dim = 2048
        B = len(prompt_ids)

        padded = torch.zeros(B, max_len, dim, dtype=self.dtype, device=self.device)
        masks = torch.zeros(B, max_len, dtype=self.dtype, device=self.device)

        for i, pid in enumerate(prompt_ids):
            if pid not in cache:
                raise KeyError(f"Item id '{pid}' not found in T5 embedding cache.")
            emb = cache[pid]          # (1, seq_len, 2048)
            seq_len = min(emb.shape[1], max_len)
            # left-pad: real tokens go at the END (LlamaGen convention)
            padded[i, -seq_len:] = emb[0, :seq_len].to(device=self.device, dtype=self.dtype)
            masks[i, -seq_len:] = 1.0

        c_indices = padded * masks[:, :, None]
        return c_indices, masks

    def train_supervised_step(self, batch: list) -> dict:
        """
        Fine-tune on image-caption pairs using AR next-token prediction loss.
        Each item in batch must have image_path and caption/prompt.
        """
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        images = []
        captions = []
        for item in batch:
            if item.image_path and Path(item.image_path).exists():
                img = Image.open(item.image_path).convert("RGB")
                images.append(transform(img))
            captions.append(item.text)

        if not images:
            # prompt-only items: no training signal, return zero loss
            return {"loss": 0.0, "lr": self.learning_rate, "grad_norm": 0.0}

        img_tensor = torch.stack(images).to(self.device)

        # Encode images to VQ tokens
        with torch.no_grad():
            _, _, [_, _, image_tokens] = self.vq_model.encode(img_tensor)

        # Get text embeddings
        with torch.no_grad():
            caption_embs, emb_masks = self.t5.get_text_embeddings(captions[:len(images)])
            new_caption_embs = []
            for emb, mask in zip(caption_embs, emb_masks):
                valid = int(mask.sum().item())
                new_caption_embs.append(torch.cat([emb[valid:], emb[:valid]]))
            new_caption_embs = torch.stack(new_caption_embs)
            new_emb_masks = torch.flip(emb_masks, dims=[-1])

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks

        # image_tokens: (B, H*W)
        B = image_tokens.shape[0]
        seq_len = image_tokens.shape[1]
        targets = image_tokens.long()  # (B, seq_len)

        self.gpt.train()
        self._optimizer.zero_grad()

        use_amp = self.precision in ("bf16", "fp16")
        amp_dtype = self.dtype if use_amp else None

        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits, _ = self.gpt(
                idx=targets[:, :-1],
                cond_idx=c_indices,
                input_pos=None,
                targets=targets,
                mask=None,
                valid=None,
            )
            # cross-entropy over image token predictions
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.gpt.parameters() if p.requires_grad],
                self.max_grad_norm,
            ).item()
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.gpt.parameters() if p.requires_grad],
                self.max_grad_norm,
            ).item()
            self._optimizer.step()

        self._step += 1
        return {
            "loss": loss.item(),
            "lr": self._optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "step": self._step,
        }

    def _get_conditioning(self, batch, t5_cache=None):
        """Returns (c_indices, c_emb_masks) for a batch, using cache if available."""
        prompts = [item.text for item in batch]
        ids = [item.id for item in batch]
        bucket = batch[0].bucket if batch else None

        if t5_cache is not None and bucket is not None:
            cache_dict = t5_cache.bucket_embeddings(bucket)
            if all(i in cache_dict for i in ids):
                return self._pack_cached_embeddings(ids, cache_dict)

        caption_embs, emb_masks = self.t5.get_text_embeddings(prompts)
        new_embs = []
        for emb, mask in zip(caption_embs, emb_masks):
            valid = int(mask.sum().item())
            new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
        c_indices = torch.stack(new_embs) * torch.flip(emb_masks, dims=[-1])[:, :, None]
        c_emb_masks = torch.flip(emb_masks, dims=[-1])
        return c_indices, c_emb_masks

    def _disable_kv_cache(self):
        """Clear KV cache set up by generate() so training forward pass works."""
        for block in self.gpt.layers:
            block.attention.kv_cache = None

    def _compute_log_probs(self, image_tokens, c_indices, c_emb_masks, cfg_scale: float = 1.0, reduction: Optional[str] = None):
        """
        Full forward pass → sequence log prob for each sequence.
        image_tokens: (B, seq_len) int64
        reduction: "mean" | "sum" | "sum_sqrt_len" (default: self.logprob_reduction)
        Returns: (B,) float tensor
        """
        import torch.nn.functional as F
        # Must stay in train() mode: LlamaGen uses self.training to select freqs_cis slicing
        # (train → [:seq_len], eval → [input_pos]). eval() with input_pos=None gives wrong shape.
        # Instead, disable only the stochastic ops that cause NaN gradients:
        #  1. CaptionEmbedder conditioning dropout (class_dropout_prob=0.1): randomly replaces
        #     T5 embeddings with uncond_embedding (a buffer with requires_grad=True), routing
        #     gradients through an inconsistent path and producing NaN.
        #  2. All nn.Dropout modules (token/residual/FFN dropout at p=0.1): stochastic scaling
        #     creates position-specific gradient magnitudes that overflow in some layers.
        # This matches the generation distribution (generated in no-dropout mode) and gives
        # clean, deterministic gradients. Restored unconditionally via try/finally.
        self.gpt.train()
        saved_uncond_prob = self.gpt.cls_embedding.uncond_prob
        self.gpt.cls_embedding.uncond_prob = 0.0
        dropout_mods = [m for m in self.gpt.modules() if isinstance(m, nn.Dropout)]
        for m in dropout_mods:
            m.eval()

        try:
            tokens_long = image_tokens.long()
            B = tokens_long.shape[0]
            # model is in float32 when this is called (cast by train_grpo_step)
            c_cast = c_indices.float()

            if cfg_scale > 1.0:
                c_uncond = torch.zeros_like(c_cast)
                tokens_2x = tokens_long.repeat(2, 1)
                c_2x = torch.cat([c_cast, c_uncond], dim=0)
                logits_2x, _ = self.gpt(
                    idx=tokens_2x[:, :-1],
                    cond_idx=c_2x,
                    input_pos=None,
                    targets=None,
                    mask=None,
                    valid=None,
                )
                logits_cond = logits_2x[:B].float().nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
                logits_uncond = logits_2x[B:].float().nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits, _ = self.gpt(
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
            token_lp = token_lp.clamp(min=-20.0)

            red = reduction if reduction is not None else self.logprob_reduction
            import math as _math
            if red == "sum":
                return token_lp.sum(dim=-1)
            elif red == "sum_sqrt_len":
                return token_lp.sum(dim=-1) / _math.sqrt(token_lp.shape[-1])
            else:  # mean
                return token_lp.mean(dim=-1)

        finally:
            self.gpt.cls_embedding.uncond_prob = saved_uncond_prob
            for m in dropout_mods:
                m.train()

    def _compute_log_probs_ref(self, image_tokens, c_indices, c_emb_masks, cfg_scale: float = 1.0):
        """Log probs under reference model (LoRA zeroed = base model)."""
        saved = {}
        for name, mod in self.gpt.named_modules():
            if hasattr(mod, "lora_B"):
                saved[name] = mod.lora_B.weight.data.clone()
                mod.lora_B.weight.data.zero_()
        with torch.no_grad():
            lp = self._compute_log_probs(image_tokens, c_indices, c_emb_masks, cfg_scale=cfg_scale)
        for name, mod in self.gpt.named_modules():
            if name in saved:
                mod.lora_B.weight.data.copy_(saved[name])
        return lp

    def train_grpo_step(
        self,
        batch: list,
        reward_model,
        num_samples: int = 4,
        beta: float = 0.01,
        t5_cache=None,
        reward_mode: str = "hard_target",
        advantage_eps: float = 1e-8,
    ) -> dict:
        """
        GRPO: generate num_samples images per prompt in memory (no disk I/O),
        score with reward_model, update policy via group-relative policy gradient.
        """
        import torch.nn.functional as F
        from autoregressive.models.generate import generate

        B = len(batch)

        # 1. Text conditioning (shared across all samples)
        with torch.no_grad():
            c_indices, c_emb_masks = self._get_conditioning(batch, t5_cache)

        # 2. Generate num_samples token sequences per prompt (no grad)
        self.gpt.eval()
        all_tokens = []   # list of (B, seq_len) tensors, length = num_samples
        all_pil_imgs = [] # flat list: [b0_s0, b1_s0, ..., bB_s0, b0_s1, ...]

        qzshape = [B, self.codebook_embed_dim, self.latent_size, self.latent_size]

        with torch.no_grad():
            for s in range(num_samples):
                index_sample = generate(
                    self.gpt, c_indices, self.latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=self.cfg_scale_train,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    sample_logits=True,
                )  # (B, seq_len)
                all_tokens.append(index_sample)

                decoded = self.vq_model.decode_code(index_sample, qzshape)
                # convert to PIL in memory — no disk I/O
                import torchvision.transforms.functional as TF
                for i in range(B):
                    img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2  # (C,H,W) in [0,1]
                    all_pil_imgs.append(TF.to_pil_image(img_t.cpu()))

        # disable KV cache set up by generate() before training forward pass
        self._disable_kv_cache()

        # 3. Score all images with reward model (PIL, no file paths)
        # Also compute hard_target score from same VLM call for reward alignment logging.
        rewards = torch.zeros(B, num_samples)
        self._last_sample_details = []  # captured for reward_details.jsonl logging
        for s in range(num_samples):
            for i, item in enumerate(batch):
                pil_img = all_pil_imgs[s * B + i]
                soft_result = reward_model.score_image(pil_img, item, mode=reward_mode)
                soft_score = soft_result["score"]
                rewards[i, s] = soft_score
                # compute hard score from same answers (no extra VLM call if possible)
                hard_result = reward_model.score_image(pil_img, item, mode="hard_target")
                self._last_sample_details.append({
                    "prompt_id": item.id,
                    "prompt": item.text,
                    "bucket": item.bucket,
                    "sample": s,
                    "soft_reward": float(soft_score),
                    "hard_reward": float(hard_result["score"]),
                    "has_uncertain": any(
                        q.get("answer", "") == "uncertain"
                        for q in soft_result.get("question_scores", [])
                    ),
                    "question_scores": soft_result.get("question_scores", []),
                })

        # 4. Group-relative advantages per prompt
        mean_r = rewards.mean(dim=1, keepdim=True)   # (B, 1)
        std_r = rewards.std(dim=1, keepdim=True) + advantage_eps
        advantages = ((rewards - mean_r) / std_r).nan_to_num(nan=0.0)  # (B, num_samples)

        # 5. Stack tokens: (B * num_samples, seq_len)
        stacked_tokens = torch.stack(all_tokens, dim=1).reshape(B * num_samples, -1)
        rep_c = c_indices.repeat_interleave(num_samples, dim=0)
        rep_masks = c_emb_masks.repeat_interleave(num_samples, dim=0)
        flat_advantages = advantages.reshape(-1).to(self.device)  # (B * num_samples,)

        # 6. KL reference log probs (no grad, base model)
        # Cast to float32 first: _compute_log_probs hardcodes c_indices.float() for conditioning,
        # so the model must also be float32 to avoid a dtype mismatch in F.linear.
        self.gpt.float()
        kl_loss = torch.tensor(0.0, device=self.device)
        ref_log_probs = None
        if beta > 0.0:
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs_ref(stacked_tokens, rep_c, rep_masks, cfg_scale=1.0)

        # 7. Chunked gradient accumulation — avoids OOM from full B*G forward with grad
        # Log probs use conditional-only forward (cfg_scale=1.0): zero unconditional conditioning
        # causes NaN intermediate activations in cls_embedding which corrupt the backward.
        # Tokens were generated with CFG for quality; log probs under conditional model
        # are still a valid policy-gradient signal (REINFORCE with baseline).
        _lp_cfg_scale = 1.0

        chunk_size = B  # process one prompt's samples at a time
        total = B * num_samples
        self._optimizer.zero_grad()
        total_pg_loss = 0.0
        total_kl = 0.0
        all_seq_lp: list = []

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            c_tok = stacked_tokens[start:end]
            c_cond = rep_c[start:end]
            c_mask = rep_masks[start:end]
            c_adv = flat_advantages[start:end]
            weight = (end - start) / total  # normalise so gradients sum correctly

            lp = self._compute_log_probs(c_tok, c_cond, c_mask, cfg_scale=_lp_cfg_scale)
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
            # Sanitize after every chunk: once a param's .grad contains NaN,
            # all subsequent += accumulations stay NaN (NaN + x = NaN in IEEE 754).
            for p in self.gpt.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # LoRA grad norm (post-NaN-sanitize, pre-clip — these are what actually get applied)
        lora_params = [p for n, p in self.gpt.named_parameters()
                       if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
        lora_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in lora_params if p.grad is not None) ** 0.5
            if lora_params else 0.0
        )

        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            [p for p in self.gpt.parameters() if p.requires_grad],
            self.max_grad_norm,
        ).item()

        self._optimizer.step()

        # LoRA weight norm (post-step)
        lora_weight_norm = (
            sum(p.data.float().norm().item() ** 2 for p in lora_params) ** 0.5
            if lora_params else 0.0
        )

        # Cast back to bfloat16 for generation
        self.gpt.to(dtype=self.dtype)

        # --- diagnostics ---
        reward_stds = rewards.std(dim=1).cpu()                       # (B,)
        flat_adv = advantages.abs().reshape(-1).cpu()
        import statistics as _stats
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
            "grad_norm": grad_norm_before_clip,
            "grad_norm_before_clip": grad_norm_before_clip,
            "grad_norm_after_clip": min(grad_norm_before_clip, self.max_grad_norm),
            "lora_weight_norm": lora_weight_norm,
            "lora_grad_norm": lora_grad_norm,
            "percent_groups_zero_std": float((reward_stds < 1e-6).float().mean().item() * 100),
            "mean_group_reward_std": float(reward_stds.mean().item()),
            "median_group_reward_std": float(_stats.median(reward_stds.tolist())),
            "mean_abs_advantage": float(flat_adv.mean().item()),
            "fraction_nonzero_advantage": float((flat_adv > 1e-6).float().mean().item()),
            "seq_logprob_mean": float(seq_lp_t.mean().item()),
            "seq_logprob_std": float(seq_lp_t.std().item()) if len(seq_lp_t) > 1 else 0.0,
            "cfg_scale_train": self.cfg_scale_train,
            "logprob_reduction": self.logprob_reduction,
            "reward_mode": reward_mode,
        }

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.use_lora:
            # save only LoRA weights (~10MB vs ~3GB for full model)
            lora_state = {
                k: v for k, v in self.gpt.state_dict().items()
                if "lora_A" in k or "lora_B" in k
            }
        else:
            lora_state = self.gpt.state_dict()
        state = {
            "gpt_model": lora_state,
            "optimizer": self._optimizer.state_dict() if self._optimizer else None,
            "step": self._step,
            "config": {
                "gpt_model": self.gpt_model_name,
                "image_size": self.image_size,
                "use_lora": self.use_lora,
                "lora_config": self.lora_config,
            },
        }
        torch.save(state, path)
        print(f"[LlamaGenWrapper] Saved checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.gpt.load_state_dict(ckpt["gpt_model"], strict=False)
        if self._optimizer and ckpt.get("optimizer"):
            self._optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
        print(f"[LlamaGenWrapper] Loaded checkpoint from {path} (step={self._step})")

    def current_checkpoint_id(self) -> str:
        return f"step_{self._step:06d}"
