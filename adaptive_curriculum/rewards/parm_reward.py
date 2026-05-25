"""
PARMRewardModel — wraps ImageSelector from Image-Generation-CoT.

Implements the same RewardModel interface used by Qwen-GRPO so that
LlamaGenWrapper.train_grpo_step() works without modification.

Key design notes:
- PARM has one scoring mode; the `mode` argument is accepted but ignored.
- Reward = parm_norm_yes_prob = yes_prob / (yes_prob + no_prob).
  This is denser than raw yes_prob (which is over a ~150K vocab denominator)
  but in practice yes+no ≈ 1.0 for this fine-tuned checkpoint, so both values
  are nearly identical. We use norm for robustness.
- Internal PIL image cache (keyed by id(image)) avoids the double PARM call
  that train_grpo_step makes per image (soft call + hard_target call).
  Call clear_cache() after each training step.
- score_images_batch(pairs, mode) is also implemented for compatibility with
  the run_val helper in GRPO/train.py which calls that signature.
"""
import copy
import logging
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from adaptive_curriculum.reward.reward_schema import RewardModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flash-attn patch (must run before ImageSelector is imported)
# ---------------------------------------------------------------------------

def _patch_flash_attn_if_missing():
    """
    Transformers auto-detects flash-attn support and raises ImportError if it
    is not installed. Monkey-patch to force eager attention instead.
    Only activates when flash_attn is truly absent; no-op otherwise.
    """
    try:
        import flash_attn  # noqa: F401
        return  # installed — no patch needed
    except ImportError:
        pass

    try:
        import transformers.modeling_utils as _tmu

        @classmethod
        def _eager_attn_only(cls, config, *args, **kwargs):
            config._attn_implementation = "eager"
            config._attn_implementation_internal = "eager"

        _tmu.PreTrainedModel._check_and_enable_flash_attn_2 = _eager_attn_only
        logger.info("[PARM] flash_attn not found — patched transformers to use eager attention")
    except Exception as e:
        logger.warning(f"[PARM] Could not patch flash_attn check: {e}")


# ---------------------------------------------------------------------------
# PARMRewardModel
# ---------------------------------------------------------------------------

class PARMRewardModel(RewardModel):
    """
    Frozen PARM verifier used as reward model for GRPO training of LlamaGen.

    Usage:
        reward = PARMRewardModel(parm_repo="/path/to/Image-Generation-CoT",
                                 parm_ckpt="/path/to/.../parm")
        result = reward.score_image(pil_image, item)
        # result["score"] == parm_norm_yes_prob in [0, 1]

    After each train_grpo_step, call reward.clear_cache() to release PIL refs.
    """

    def __init__(
        self,
        parm_repo: str,
        parm_ckpt: str,
        device: str = "cuda",
        device_map: str = "single",
        score_batch_size: int = 4,
    ):
        """
        parm_repo       — path to Image-Generation-CoT repo root
        parm_ckpt       — path to PARM checkpoint folder (contains config.json etc.)
        device          — torch device string, e.g. "cuda"
        device_map      — "single" (GPU 0 only) or "auto" (multi-GPU via accelerate)
        score_batch_size — images per PARM forward pass (only effective when all
                           images share the same prompt — true GPU batching)
        """
        self.parm_repo = parm_repo
        self.parm_ckpt = parm_ckpt
        self.device = device
        self.score_batch_size = score_batch_size

        _patch_flash_attn_if_missing()

        if parm_repo not in sys.path:
            sys.path.insert(0, parm_repo)

        from selector import ImageSelector  # noqa: E402  (lives in parm_repo)

        device_map_arg = {"": 0} if device_map == "single" else "auto"
        logger.info(f"[PARM] Loading checkpoint from {parm_ckpt} ...")
        self.selector = ImageSelector(
            pretrained=parm_ckpt,
            device=device,
            device_map=device_map_arg,
        )
        self.selector.model.eval()
        for p in self.selector.model.parameters():
            p.requires_grad = False

        self.yes_id = self.selector.tokenizer.convert_tokens_to_ids("yes")
        self.no_id  = self.selector.tokenizer.convert_tokens_to_ids("no")

        # Cache PIL results within one train_grpo_step to avoid the double call
        # (train_grpo_step calls score_image once for the reward + once for hard_target
        #  on the SAME pil_img object — caching by id() makes the second call free).
        self._cache: Dict[int, dict] = {}

        n_params = sum(p.numel() for p in self.selector.model.parameters())
        logger.info(f"[PARM] Ready. Parameters: {n_params / 1e9:.2f}B  "
                    f"yes_id={self.yes_id}  no_id={self.no_id}")

    # ------------------------------------------------------------------
    # RewardModel interface
    # ------------------------------------------------------------------

    def score_image(self, image, item, mode: str = "hard_target") -> dict:
        """
        image : PIL.Image.Image  (in-memory, no disk I/O needed)
        item  : any object with .text or .prompt attribute
        mode  : ignored — PARM has one scoring mode
        Returns dict with "score" == parm_norm_yes_prob in [0, 1].
        """
        cache_key = id(image)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = _get_prompt(item)
        try:
            result = self._score_single(image, prompt)
        except Exception as exc:
            logger.warning(f"[PARM] score_image failed: {exc}")
            result = _zero_result(error=str(exc))

        self._cache[cache_key] = result
        return result

    def score_batch(
        self,
        images: List,
        items: list,
        mode: str = "hard_target",
    ) -> List[dict]:
        """Sequential scoring of a mixed (prompt, image) list."""
        return [self.score_image(img, item, mode=mode) for img, item in zip(images, items)]

    def score_images_batch(
        self,
        pairs: List[Tuple],
        mode: str = "hard_target",
    ) -> List[dict]:
        """
        Compatible with the signature used in GRPO/train.py's run_val helper:
            reward_model.score_images_batch(list_of_(image, item)_tuples, mode=...)
        """
        return [self.score_image(img, item, mode=mode) for img, item in pairs]

    def clear_cache(self):
        """Release cached PIL image references. Call after each train_grpo_step."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Grouped batch scoring (for probe script / eval — same prompt)
    # ------------------------------------------------------------------

    def score_grouped(
        self,
        prompt: str,
        pil_images: List,
        chunk_size: Optional[int] = None,
    ) -> List[dict]:
        """
        Score all images for ONE prompt in true batched forward passes.
        Same prompt → same input_ids → true GPU batching.

        Use this from the probe script or eval harness, not from inside
        train_grpo_step (which calls score_image per image for compatibility).
        """
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        chunk_size = chunk_size or self.score_batch_size

        question = (
            f"{DEFAULT_IMAGE_TOKEN} This image is generated by a prompt: {prompt}. "
            f"Does this image accurately represent the prompt? "
            f"Please answer yes or no without explanation."
        )
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids_single = tokenizer_image_token(
            prompt_question, self.selector.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(self.device)

        all_results = []
        for start in range(0, len(pil_images), chunk_size):
            chunk = pil_images[start: start + chunk_size]
            try:
                image_tensors = process_images(
                    chunk, self.selector.image_processor, self.selector.model.config
                )
                if isinstance(image_tensors, torch.Tensor):
                    image_tensors = image_tensors.to(dtype=torch.float16, device=self.device)
                    img_list = [image_tensors[i] for i in range(len(chunk))]
                else:
                    img_list = [t.to(dtype=torch.float16, device=self.device) for t in image_tensors]

                B = len(chunk)
                input_ids = input_ids_single.unsqueeze(0).repeat(B, 1)

                with torch.no_grad():
                    cont = self.selector.model.generate(
                        input_ids,
                        images=img_list,
                        image_sizes=[img.size for img in chunk],
                        do_sample=True,
                        temperature=1.0,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                probs    = F.softmax(cont.scores[0], dim=-1)   # (B, vocab)
                yes_probs = probs[:, self.yes_id].tolist()
                no_probs  = probs[:, self.no_id].tolist()
                gen_ids   = cont.sequences[:, 0].tolist()      # LLaVA returns only generated tokens

                for yp, np_, gen_id in zip(yes_probs, no_probs, gen_ids):
                    resp = self.selector.tokenizer.convert_ids_to_tokens([gen_id])[0].lower()
                    all_results.append(_make_result(yp, np_, resp))

            except Exception as exc:
                logger.warning(f"[PARM] Batch chunk failed ({exc}), falling back to sequential")
                for img in chunk:
                    try:
                        all_results.append(self._score_single(img, prompt))
                    except Exception as exc2:
                        all_results.append(_zero_result(error=str(exc2)))

        return all_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_single(self, pil_image, prompt: str) -> dict:
        """Score one PIL image (batch-size 1 forward pass)."""
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        question = (
            f"{DEFAULT_IMAGE_TOKEN} This image is generated by a prompt: {prompt}. "
            f"Does this image accurately represent the prompt? "
            f"Please answer yes or no without explanation."
        )
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, self.selector.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        image_tensors = process_images(
            [pil_image], self.selector.image_processor, self.selector.model.config
        )
        if isinstance(image_tensors, torch.Tensor):
            image_tensors = image_tensors.to(dtype=torch.float16, device=self.device)
            img_list = [image_tensors[0]]
        else:
            img_list = [image_tensors[0].to(dtype=torch.float16, device=self.device)]

        with torch.no_grad():
            cont = self.selector.model.generate(
                input_ids,
                images=img_list,
                image_sizes=[pil_image.size],
                do_sample=True,
                temperature=1.0,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

        probs   = F.softmax(cont.scores[0], dim=-1)   # (1, vocab)
        yes_prob = probs[0, self.yes_id].item()
        no_prob  = probs[0, self.no_id].item()
        gen_id   = cont.sequences[0, 0].item()        # LLaVA returns only generated tokens
        resp     = self.selector.tokenizer.convert_ids_to_tokens([gen_id])[0].lower()

        return _make_result(yes_prob, no_prob, resp)


# ---------------------------------------------------------------------------
# Pure helpers (module-level, no class state)
# ---------------------------------------------------------------------------

def _get_prompt(item) -> str:
    return getattr(item, "text", None) or getattr(item, "prompt", "") or ""


def _make_result(yes_prob: float, no_prob: float, resp: str) -> dict:
    norm_yes = yes_prob / (yes_prob + no_prob + 1e-8)
    return {
        "score": norm_yes,
        "question_scores": [],
        "component_scores": {
            "parm_yes_prob":      yes_prob,
            "parm_no_prob":       no_prob,
            "parm_norm_yes_prob": norm_yes,
            "parm_selected":      float(resp == "yes"),
        },
        "mode": "parm_norm_yes_prob",
    }


def _zero_result(error: str = "") -> dict:
    return {
        "score": 0.0,
        "question_scores": [],
        "component_scores": {
            "parm_yes_prob":      0.0,
            "parm_no_prob":       0.0,
            "parm_norm_yes_prob": 0.0,
            "parm_selected":      0.0,
            "parm_error":         error,
        },
        "mode": "parm_norm_yes_prob",
    }
