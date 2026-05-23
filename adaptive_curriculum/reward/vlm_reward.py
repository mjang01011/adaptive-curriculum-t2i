"""
VLM-based compositional reward using local Qwen3-VL.

Key design: one VLM call per image answers ALL target questions at once,
reducing inference cost by 2-4× vs one call per question.

Reward modes
------------
hard_target                   : correct yes/no = 1, incorrect or uncertain = 0  (UCB/eval signal)
                                 Uses item.target_questions only.
pseudo_soft_target             : correct = 1, uncertain = 0.5, incorrect = 0     (GRPO signal, legacy)
                                 Uses item.target_questions only.
pseudo_soft_grpo               : weighted pseudo-soft scoring on item.grpo_reward_questions.
                                 score = sum(weight_i * pseudo_soft_i) / sum(weight_i)
                                 Falls back to pseudo_soft_target on target_questions if no grpo_reward_questions.
pseudo_soft_grpo_target_heavy  : same as pseudo_soft_grpo but overrides per-question weights using
                                 _TARGET_HEAVY_BASE_WEIGHTS (relation/count=6×, attribute=3×, etc.)
true_soft                      : score = P_vlm("yes") if expected="yes" else 1 - P_vlm("yes"),
                                 derived from first-token logits — one call per question (slower)
qwen_logit_grpo_target_heavy   : logit-based continuous score on item.grpo_reward_questions with
                                 target-heavy weighting. One VLM call per question (like true_soft).
                                 score_i = p_yes/(p_yes+p_no) if expected="yes" else p_no/(p_yes+p_no)
                                 Uses same _TARGET_HEAVY_BASE_WEIGHTS as pseudo_soft_grpo_target_heavy.
                                 Slower but denser signal than pseudo-soft modes.
"""
import json
import math
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from adaptive_curriculum.reward.reward_schema import RewardModel
from adaptive_curriculum.reward.grpo_rewards import GATED_V2_MODES, apply_gated_v2


# ---------------------------------------------------------------------------
# Prompt builder and response parser (module-level, testable independently)
# ---------------------------------------------------------------------------

def build_vlm_question_prompt(target_questions: list) -> str:
    """Build a multi-question prompt that asks Qwen to return all answers as JSON."""
    lines = [
        "You are evaluating whether an image satisfies a list of visual questions.",
        "",
        "Answer each question with exactly one of:",
        "yes, no, uncertain",
        "",
        "Return only valid JSON in this format:",
        '{',
        '  "answers": [',
        '    {"id": 0, "answer": "yes"},',
        '    {"id": 1, "answer": "no"}',
        '  ]',
        '}',
        "",
        "Questions:",
    ]
    for i, q in enumerate(target_questions):
        text = q.question if hasattr(q, "question") else q["question"]
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def parse_vlm_json_answers(response_text: str, n_questions: int) -> Optional[List[str]]:
    """
    Parse Qwen JSON response into a list of n_questions answers.
    Returns None if JSON cannot be parsed (triggers retry in caller).
    Invalid answers are mapped to 'uncertain'.
    """
    valid = {"yes", "no", "uncertain"}
    text = response_text.strip()
    data = None

    # 1. Direct parse
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown code fence
    if data is None:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

    # 3. Grab the first {...} block in the response (handles leading explanation text)
    if data is None:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

    if data is None:
        return None

    answers_raw = data.get("answers", [])
    if not isinstance(answers_raw, list) or len(answers_raw) == 0:
        return None

    answers: List[str] = []
    for entry in answers_raw:
        if isinstance(entry, dict):
            ans = str(entry.get("answer", "uncertain")).strip().lower()
        else:
            ans = str(entry).strip().lower()
        answers.append(ans if ans in valid else "uncertain")

    # Pad to n_questions if response was truncated
    while len(answers) < n_questions:
        answers.append("uncertain")

    return answers[:n_questions]


def _hard_score(predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def _pseudo_soft_score(predicted: str, expected: str) -> float:
    if predicted == "uncertain":
        return 0.5
    return 1.0 if predicted == expected else 0.0


# Base unnormalized weights per question type for target-heavy mode.
# Target questions (relation/attribute/count) get 6×; anti-target get 1.5×;
# presence/support get 0.75×; quality/alignment get 0.5×.
# Weights are normalized per-item so they always sum to 1.0.
_TARGET_HEAVY_BASE_WEIGHTS = {
    "relation":        6.0,
    "attribute":       3.0,
    "count":           6.0,
    "anti_relation":   1.5,
    "anti_swap":       0.75,
    "anti_count":      1.5,
    "object_presence": 0.75,
    "image_quality":   0.5,
    "prompt_alignment": 0.5,
}
_TARGET_HEAVY_DEFAULT_WEIGHT = 1.0


def _target_heavy_weight(q_type: str) -> float:
    return _TARGET_HEAVY_BASE_WEIGHTS.get(q_type, _TARGET_HEAVY_DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Reward model class
# ---------------------------------------------------------------------------

class Qwen3VLRewardModel(RewardModel):
    """
    Runs Qwen3-VL locally. No API key needed.
    Uses one VLM call per image to answer all target questions (one-pass scoring).
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        # efficiency counters
        self._total_vlm_calls = 0
        self._total_questions_answered = 0
        self._total_vlm_seconds = 0.0

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self._model.eval()

    def _build_inputs(self, image, prompt_text: str):
        """Build model inputs from image (path str or PIL) and full prompt text."""
        from qwen_vl_utils import process_vision_info
        self._load()

        if isinstance(image, str):
            image_content = {"type": "image", "image": Path(image).as_uri()}
        else:
            image_content = {"type": "image", "image": image}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(next(self._model.parameters()).device)
        return inputs

    def _generate_text(self, inputs, max_new_tokens: int) -> str:
        import torch
        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

    def _build_batch_inputs(self, images_and_prompts: list):
        """Build processor inputs for a list of (image, prompt_text) pairs."""
        from qwen_vl_utils import process_vision_info
        self._load()
        all_texts = []
        all_image_inputs = []
        for image, prompt_text in images_and_prompts:
            if isinstance(image, str):
                image_content = {"type": "image", "image": Path(image).as_uri()}
            else:
                image_content = {"type": "image", "image": image}
            messages = [{"role": "user", "content": [
                image_content, {"type": "text", "text": prompt_text}
            ]}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            all_texts.append(text)
            all_image_inputs.extend(image_inputs)
        return self._processor(
            text=all_texts,
            images=all_image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(next(self._model.parameters()).device)

    def _generate_text_batch(self, images_and_prompts: list, max_new_tokens: int) -> list:
        """One generate() call for a batch of (image, prompt) pairs. Returns list of strings."""
        import torch
        inputs = self._build_batch_inputs(images_and_prompts)
        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        return [t.strip() for t in self._processor.batch_decode(
            out_ids[:, input_len:], skip_special_tokens=True
        )]

    def _generate_with_scores(self, inputs, max_new_tokens: int):
        """Generate and return (text, scores) for first-token probability extraction."""
        import torch
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        text = self._processor.batch_decode(
            out.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        return text, out.scores

    @staticmethod
    def _parse_yes_no(raw: str) -> str:
        raw = raw.strip().lower()
        if "uncertain" in raw:
            return "uncertain"
        if "yes" in raw:
            return "yes"
        return "no"

    def _ask_single(self, image, question: str) -> str:
        """Single yes/no/uncertain answer for one question. Used by true_soft mode."""
        prompt = f"{question}\nAnswer with only 'yes', 'no', or 'uncertain'."
        inputs = self._build_inputs(image, prompt)
        raw = self._generate_text(inputs, max_new_tokens=8)
        return self._parse_yes_no(raw)

    def _ask_single_with_prob(self, image, question: str) -> Tuple[str, float]:
        """
        Single answer + P(yes) from next-token logits via a forward pass.
        Avoids output_scores=True in generate(), which crashes with Qwen3-VL
        when a second model (LlamaGen) is loaded concurrently on the same GPU.
        """
        import torch
        prompt = f"{question}\nAnswer with only 'yes', 'no', or 'uncertain'."
        inputs = self._build_inputs(image, prompt)

        with torch.no_grad():
            outputs = self._model(**inputs)
        # logits: (1, seq_len, vocab_size) — last position predicts first answer token
        last_logits = outputs.logits[0, -1, :]

        tok = self._processor.tokenizer

        def max_logit(strings):
            best = -1e9
            for s in strings:
                ids = tok.encode(s, add_special_tokens=False)
                if ids:
                    val = last_logits[ids[0]].item()
                    if val > best:
                        best = val
            return best

        yes_logit = max_logit(["yes", "Yes", "YES"])
        no_logit  = max_logit(["no", "No", "NO"])
        unc_logit = max_logit(["uncertain", "Uncertain"])

        p_yes = 1.0 / (1.0 + math.exp(-(yes_logit - no_logit)))

        # discrete answer from argmax of the three candidates
        if unc_logit > yes_logit and unc_logit > no_logit:
            answer = "uncertain"
        elif yes_logit >= no_logit:
            answer = "yes"
        else:
            answer = "no"

        return answer, p_yes

    def answer_all_questions_once(self, image, target_questions: list) -> List[str]:
        """
        One VLM call per image answers all target_questions simultaneously.
        Falls back to ["uncertain"] * n if JSON parsing fails after one retry.
        """
        n = len(target_questions)
        if n == 0:
            return []

        prompt = build_vlm_question_prompt(target_questions)
        inputs = self._build_inputs(image, prompt)
        raw = self._generate_text(inputs, max_new_tokens=256)
        answers = parse_vlm_json_answers(raw, n)

        if answers is None:
            retry_prompt = prompt + "\n\nRespond ONLY with valid JSON. No additional text."
            inputs2 = self._build_inputs(image, retry_prompt)
            raw2 = self._generate_text(inputs2, max_new_tokens=256)
            answers = parse_vlm_json_answers(raw2, n)

        return answers if answers is not None else ["uncertain"] * n

    def answer_all_questions_once_batch(self, images_and_question_lists: list) -> list:
        """
        One generate() call for a batch of (image, questions_list) pairs.
        Returns list of answer-lists, one per image.
        Any image whose JSON fails to parse is retried individually.
        """
        if not images_and_question_lists:
            return []
        n_list = [len(qs) for _, qs in images_and_question_lists]
        prompts = [build_vlm_question_prompt(qs) for _, qs in images_and_question_lists]
        max_tokens = max(max(n_list) * 50, 256)

        raw_texts = self._generate_text_batch(
            [(img, p) for (img, _), p in zip(images_and_question_lists, prompts)],
            max_new_tokens=max_tokens,
        )

        results = []
        for raw, n, (img, qs), prompt in zip(
            raw_texts, n_list, images_and_question_lists, prompts
        ):
            answers = parse_vlm_json_answers(raw, n)
            if answers is None:
                retry = prompt + "\n\nRespond ONLY with valid JSON. No additional text."
                inputs2 = self._build_inputs(img, retry)
                raw2 = self._generate_text(inputs2, max_new_tokens=max_tokens)
                answers = parse_vlm_json_answers(raw2, n)
            results.append(answers if answers is not None else ["uncertain"] * n)
        return results

    def score_images_batch(self, images_and_items: list, mode: str) -> list:
        """
        Score a batch of (image, item) pairs with one VLM generate() call.
        Returns a list of score dicts in the same order as input.

        Logit-based modes (true_soft, qwen_logit_grpo_target_heavy) fall back
        to sequential score_image() since they need per-question logits.
        """
        if not images_and_items:
            return []

        # Logit modes can't batch easily — fall back
        if mode in ("true_soft", "qwen_logit_grpo_target_heavy"):
            return [self.score_image(img, item, mode=mode)
                    for img, item in images_and_items]

        # Select question set per item
        def _get_questions(item):
            if mode in ("pseudo_soft_grpo", "pseudo_soft_grpo_target_heavy") \
                    or mode in GATED_V2_MODES:
                qs = getattr(item, "grpo_reward_questions", []) or []
                return qs if qs else item.target_questions
            return item.target_questions

        questions_list = [_get_questions(item) for _, item in images_and_items]

        t0 = time.time()
        answers_list = self.answer_all_questions_once_batch(
            [(img, qs) for (img, _), qs in zip(images_and_items, questions_list)]
        )
        vlm_seconds = time.time() - t0
        per_img_s = vlm_seconds / len(images_and_items)

        self._total_vlm_calls += 1
        self._total_questions_answered += sum(len(qs) for qs in questions_list)
        self._total_vlm_seconds += vlm_seconds

        results = []
        for (_, item), questions, answers in zip(
            images_and_items, questions_list, answers_list
        ):
            n = len(questions)

            # Build per-question weights (same logic as score_image)
            if mode in ("pseudo_soft_grpo_target_heavy",):
                raw_w = [_target_heavy_weight(getattr(q, "q_type", "")) for q in questions]
                total = sum(raw_w) or 1.0
                eff_w = [w / total for w in raw_w]
            elif mode == "pseudo_soft_grpo":
                raw_w = [getattr(q, "weight", 1.0) for q in questions]
                total = sum(raw_w) or 1.0
                eff_w = [w / total for w in raw_w]
            else:
                eff_w = [1.0 / n] * n if n > 0 else []

            question_scores = []
            for pred, q, w in zip(answers, questions, eff_w):
                expected = q.answer.lower()
                q_score = _pseudo_soft_score(pred, expected) \
                    if mode != "hard_target" else _hard_score(pred, expected)
                question_scores.append({
                    "question": q.question, "expected": expected,
                    "predicted": pred, "correct": pred == expected,
                    "score": q_score, "weight": w,
                    "q_type": getattr(q, "q_type", "unknown"),
                })

            scalar = sum(qs["score"] * qs["weight"] for qs in question_scores)

            # Component breakdown
            comps: dict = {}
            for qs in question_scores:
                qt = qs["q_type"]
                comps.setdefault(qt, {"s": 0.0, "w": 0.0})
                comps[qt]["s"] += qs["score"] * qs["weight"]
                comps[qt]["w"] += qs["weight"]
            component_scores = {
                qt: v["s"] / v["w"] if v["w"] > 0 else 0.0
                for qt, v in comps.items()
            }

            # Gated-v2 modes: override scalar with formula
            if mode in GATED_V2_MODES:
                type_scores: dict = {}
                type_counts: dict = {}
                n_uncertain = 0
                for pred, q in zip(answers, questions):
                    qt = getattr(q, "q_type", "unknown")
                    s = _pseudo_soft_score(pred, q.answer.lower())
                    type_scores[qt] = type_scores.get(qt, 0.0) + s
                    type_counts[qt] = type_counts.get(qt, 0) + 1
                    if pred == "uncertain":
                        n_uncertain += 1
                comp = {qt: type_scores[qt] / type_counts[qt] for qt in type_scores}
                comp["uncertain_frac"] = n_uncertain / max(n, 1)
                bucket = getattr(item, "bucket", "")
                reward, debug = apply_gated_v2(mode, comp, bucket=bucket)
                qs_v2 = [{
                    "question": q.question, "expected": q.answer.lower(),
                    "predicted": pred, "correct": pred == q.answer.lower(),
                    "score": _pseudo_soft_score(pred, q.answer.lower()),
                    "weight": 1.0, "q_type": getattr(q, "q_type", "unknown"),
                } for pred, q in zip(answers, questions)]
                results.append({
                    "score": float(reward),
                    "question_scores": qs_v2,
                    "component_scores": comp,
                    "mode": mode,
                    "vlm_seconds": per_img_s,
                    "num_questions": n,
                    "reward_debug": {**debug, "reward_mode": mode},
                })
            else:
                results.append({
                    "score": float(scalar),
                    "question_scores": question_scores,
                    "component_scores": component_scores,
                    "mode": mode,
                    "vlm_seconds": per_img_s,
                    "num_questions": n,
                })

        return results

    def score_image(self, image, item, mode: str = "hard_target") -> dict:
        # Select question set based on mode
        if mode in ("pseudo_soft_grpo", "pseudo_soft_grpo_target_heavy",
                    "qwen_logit_grpo_target_heavy") or mode in GATED_V2_MODES:
            questions = getattr(item, "grpo_reward_questions", []) or []
            if not questions:
                questions = item.target_questions
        else:
            questions = item.target_questions

        n = len(questions)
        t0 = time.time()

        if mode in ("true_soft", "qwen_logit_grpo_target_heavy"):
            answers = []
            p_yes_list = []
            for q in questions:
                text = q.question if hasattr(q, "question") else q["question"]
                ans, p_yes = self._ask_single_with_prob(image, text)
                answers.append(ans)
                p_yes_list.append(p_yes)
        else:
            answers = self.answer_all_questions_once(image, questions)
            p_yes_list = [None] * n

        vlm_seconds = time.time() - t0
        vlm_calls = n if mode in ("true_soft", "qwen_logit_grpo_target_heavy") else 1
        self._total_vlm_calls += vlm_calls
        self._total_questions_answered += n
        self._total_vlm_seconds += vlm_seconds

        # Build per-question weights
        if mode in ("pseudo_soft_grpo_target_heavy", "qwen_logit_grpo_target_heavy"):
            raw_weights = [_target_heavy_weight(getattr(q, "q_type", "")) for q in questions]
            total_raw = sum(raw_weights) or 1.0
            effective_weights = [w / total_raw for w in raw_weights]
        elif mode == "pseudo_soft_grpo":
            data_weights = [getattr(q, "weight", 1.0) for q in questions]
            total_data = sum(data_weights) or 1.0
            effective_weights = [w / total_data for w in data_weights]
        else:
            effective_weights = [1.0 / n] * n if n > 0 else []

        question_scores = []
        for pred, q, p_yes, eff_w in zip(answers, questions, p_yes_list, effective_weights):
            expected = q.answer.lower()

            if mode in ("true_soft", "qwen_logit_grpo_target_heavy"):
                q_score = p_yes if expected == "yes" else (1.0 - p_yes)
                correct = pred == expected
            elif mode in ("pseudo_soft_target", "pseudo_soft_grpo", "pseudo_soft_grpo_target_heavy"):
                q_score = _pseudo_soft_score(pred, expected)
                correct = pred == expected
            else:  # hard_target
                q_score = _hard_score(pred, expected)
                correct = pred == expected

            q_type = getattr(q, "q_type", "unknown")
            question_scores.append({
                "question": q.question,
                "expected": expected,
                "predicted": pred,
                "correct": correct,
                "score": q_score,
                "weight": eff_w,
                "q_type": q_type,
            })

        score = sum(qs["score"] * qs["weight"] for qs in question_scores) if question_scores else 0.0

        # Component breakdown by q_type (for logging)
        components: dict = {}
        for qs in question_scores:
            qt = qs["q_type"]
            components.setdefault(qt, {"score_sum": 0.0, "weight_sum": 0.0, "count": 0})
            components[qt]["score_sum"] += qs["score"] * qs["weight"]
            components[qt]["weight_sum"] += qs["weight"]
            components[qt]["count"] += 1
        component_scores = {
            qt: v["score_sum"] / v["weight_sum"] if v["weight_sum"] > 0 else 0.0
            for qt, v in components.items()
        }

        # ── Gated-v2 compositional reward ────────────────────────────────
        # For these modes: build a flat comp dict (one float per q_type, using
        # pseudo-soft scores), add uncertain_frac, then apply the formula.
        if mode in GATED_V2_MODES:
            # Per-question pseudo-soft scores (ignoring weights — formula handles weighting)
            type_scores: dict = {}
            type_counts: dict = {}
            n_uncertain = 0
            for pred, q in zip(answers, questions):
                expected = q.answer.lower()
                q_type = getattr(q, "q_type", "unknown")
                q_score = _pseudo_soft_score(pred, expected)
                type_scores[q_type] = type_scores.get(q_type, 0.0) + q_score
                type_counts[q_type] = type_counts.get(q_type, 0) + 1
                if pred == "uncertain":
                    n_uncertain += 1

            comp = {qt: type_scores[qt] / type_counts[qt] for qt in type_scores}
            comp["uncertain_frac"] = n_uncertain / max(n, 1)

            bucket = getattr(item, "bucket", "")
            reward, debug = apply_gated_v2(mode, comp, bucket=bucket)

            # Re-build question_scores with pseudo-soft scoring for logging consistency
            question_scores_v2 = []
            for pred, q in zip(answers, questions):
                expected = q.answer.lower()
                q_score = _pseudo_soft_score(pred, expected)
                question_scores_v2.append({
                    "question":  q.question,
                    "expected":  expected,
                    "predicted": pred,
                    "correct":   pred == expected,
                    "score":     q_score,
                    "weight":    1.0,
                    "q_type":    getattr(q, "q_type", "unknown"),
                })

            return {
                "score":           float(reward),
                "question_scores": question_scores_v2,
                "component_scores": comp,
                "mode":            mode,
                "vlm_seconds":     vlm_seconds,
                "num_questions":   n,
                "reward_debug":    {**debug, "reward_mode": mode},
            }

        return {
            "score": float(score),
            "question_scores": question_scores,
            "component_scores": component_scores,
            "mode": mode,
            "vlm_seconds": vlm_seconds,
            "num_questions": n,
        }

    def get_efficiency_metrics(self) -> dict:
        calls = self._total_vlm_calls
        questions = self._total_questions_answered
        seconds = self._total_vlm_seconds
        return {
            "num_vlm_calls": calls,
            "num_questions_answered": questions,
            "questions_per_vlm_call": questions / calls if calls > 0 else 0.0,
            "vlm_seconds_total": seconds,
            "vlm_seconds_per_image": seconds / calls if calls > 0 else 0.0,
            "vlm_seconds_per_question": seconds / questions if questions > 0 else 0.0,
        }

    def reset_efficiency_counters(self):
        self._total_vlm_calls = 0
        self._total_questions_answered = 0
        self._total_vlm_seconds = 0.0


def build_reward_model(config) -> RewardModel:
    reward_type = getattr(config.evaluation, "reward_model", "heuristic")
    if reward_type == "heuristic":
        from adaptive_curriculum.reward.heuristic_reward import HeuristicRewardModel
        return HeuristicRewardModel()
    elif reward_type == "qwen3vl":
        model_id = getattr(config.evaluation, "vlm_model", "Qwen/Qwen3-VL-4B-Instruct")
        return Qwen3VLRewardModel(model_id=model_id)
    else:
        raise ValueError(f"Unknown reward_model: {reward_type}")
