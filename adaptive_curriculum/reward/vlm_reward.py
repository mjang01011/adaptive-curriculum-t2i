"""
VLM-based compositional reward using local Qwen3-VL.

Key design: one VLM call per image answers ALL target questions at once,
reducing inference cost by 2-4× vs one call per question.

Reward modes
------------
hard_target        : correct yes/no = 1, incorrect or uncertain = 0  (UCB/eval signal)
                     Uses item.target_questions only.
pseudo_soft_target : correct = 1, uncertain = 0.5, incorrect = 0     (GRPO signal, legacy)
                     Uses item.target_questions only.
pseudo_soft_grpo   : weighted pseudo-soft scoring on item.grpo_reward_questions.
                     score = sum(weight_i * pseudo_soft_i) / sum(weight_i)
                     Falls back to pseudo_soft_target on target_questions if no grpo_reward_questions.
true_soft          : score = P_vlm("yes") if expected="yes" else 1 - P_vlm("yes"),
                     derived from first-token logits — one call per question (slower)
"""
import json
import math
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from adaptive_curriculum.reward.reward_schema import RewardModel


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
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
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
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

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
        """Single answer + P(yes) from first-token logits. Used by true_soft mode."""
        prompt = f"{question}\nAnswer with only 'yes', 'no', or 'uncertain'."
        inputs = self._build_inputs(image, prompt)
        raw, scores = self._generate_with_scores(inputs, max_new_tokens=8)
        answer = self._parse_yes_no(raw)

        p_yes = 0.5  # safe fallback
        if scores:
            scores_0 = scores[0][0]  # (vocab_size,)
            tok = self._processor.tokenizer

            def max_logit(strings):
                best = -1e9
                for s in strings:
                    ids = tok.encode(s, add_special_tokens=False)
                    if ids:
                        val = scores_0[ids[0]].item()
                        if val > best:
                            best = val
                return best

            yes_logit = max_logit(["yes", "Yes", "YES"])
            no_logit = max_logit(["no", "No", "NO"])
            p_yes = 1.0 / (1.0 + math.exp(-(yes_logit - no_logit)))

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

    def score_image(self, image, item, mode: str = "hard_target") -> dict:
        # Select question set based on mode
        if mode in ("pseudo_soft_grpo", "pseudo_soft_grpo_target_heavy"):
            questions = getattr(item, "grpo_reward_questions", []) or []
            if not questions:
                questions = item.target_questions
        else:
            questions = item.target_questions

        n = len(questions)
        t0 = time.time()

        if mode == "true_soft":
            answers = []
            p_yes_list = []
            for q in questions:
                ans, p_yes = self._ask_single_with_prob(image, q.question)
                answers.append(ans)
                p_yes_list.append(p_yes)
        else:
            answers = self.answer_all_questions_once(image, questions)
            p_yes_list = [None] * n

        vlm_seconds = time.time() - t0
        vlm_calls = n if mode == "true_soft" else 1
        self._total_vlm_calls += vlm_calls
        self._total_questions_answered += n
        self._total_vlm_seconds += vlm_seconds

        # Build per-question weights
        if mode == "pseudo_soft_grpo_target_heavy":
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

            if mode == "true_soft":
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
