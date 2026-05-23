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
from adaptive_curriculum.reward.grpo_rewards import (
    GATED_V2_MODES, CONTRASTIVE_RUBRIC_MODES,
    apply_gated_v2, apply_contrastive_rubric,
)


# ---------------------------------------------------------------------------
# Rubric prompt template (used by contrastive rubric modes)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers: parse item metadata for contrastive statement generation
# ---------------------------------------------------------------------------

def _parse_attr_pairs_from_questions(questions: list):
    """
    Extract [(obj, attr)] from attribute-type questions with expected='yes'.
    Matches pattern: "Is the {obj} {attr}?"
    Returns list of (obj_str, attr_str) in order of appearance.
    """
    import re
    pairs = []
    for q in questions:
        if getattr(q, "q_type", "") == "attribute" and q.answer.lower() == "yes":
            m = re.match(r"Is the (.+?) (\w[\w\-]*)[\?.]?$", q.question, re.IGNORECASE)
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


_RELATION_TERMS = [
    "on top of", "to the left of", "to the right of",
    "above", "below", "under", "in front of", "behind",
    "next to", "beside", "left of", "right of",
]
_RELATION_RE = None  # lazily compiled


def _parse_spatial_info_from_questions(questions: list):
    """
    Extract {obj1, relation, obj2, opposite_relation} from relation/anti_relation questions.
    Matches pattern: "Is the {obj1} {relation} the {obj2}?"
    Returns dict or None.
    """
    import re
    global _RELATION_RE
    if _RELATION_RE is None:
        _RELATION_RE = re.compile(
            r"Is the (.+?) ({}) the (.+?)[\?.]?$".format(
                "|".join(re.escape(r) for r in _RELATION_TERMS)
            ),
            re.IGNORECASE,
        )
    rel_info  = None
    opp_rel   = None
    for q in questions:
        m = _RELATION_RE.match(q.question)
        if not m:
            continue
        qt = getattr(q, "q_type", "")
        if qt == "relation" and q.answer.lower() == "yes" and rel_info is None:
            rel_info = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
        elif qt == "anti_relation" and q.answer.lower() == "no" and opp_rel is None:
            opp_rel = m.group(2).strip()
    if rel_info:
        return dict(
            obj1=rel_info[0], relation=rel_info[1], obj2=rel_info[2],
            opposite_relation=opp_rel,
        )
    return None


RUBRIC_TEMPLATE = (
    "You are an evaluator for an AI-generated text-to-image compositionality benchmark.\n\n"
    "Your task is to decide whether the image visually supports a specific statement "
    "about the original prompt.\n\n"
    "Original prompt:\n\"{prompt}\"\n\n"
    "Evaluation rules:\n"
    "1. Judge only what is visible in the image.\n"
    "2. Do not infer hidden, cropped, or missing objects.\n"
    "3. Minor blur, stylization, or imperfect rendering is acceptable only if the object "
    "and attribute/relation are still recognizable.\n"
    "4. If the relevant object is missing or not recognizable, answer \"no\" for statements "
    "about that object.\n"
    "5. If the image is too ambiguous to decide, answer \"uncertain\".\n"
    "6. Do not reward broad semantic similarity if the specific statement is false.\n"
    "7. Use \"yes\" only when the statement is clearly supported.\n"
    "8. Use \"no\" when the statement is clearly false, contradicted, or depends on a missing object.\n"
    "9. Use \"uncertain\" when the statement may be true but the image is unclear.\n\n"
    "Statement to verify:\n{statement}\n\n"
    "Answer with exactly one word: yes, no, or uncertain."
)


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

    if data is None or not isinstance(data, dict):
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
        self._processor.tokenizer.padding_side = "left"
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
            out_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,
            )
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
            out_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        return [t.strip() for t in self._processor.batch_decode(
            out_ids[:, input_len:], skip_special_tokens=True
        )]

    def _forward_probs_batch(self, image_prompt_pairs: list, chunk_size: int = 32) -> list:
        """
        Core logit-extraction method. Takes pre-formatted (image, full_prompt_text) pairs.
        Returns list of {"yes": float, "no": float, "uncertain": float, "margin": float}.
        """
        import torch
        if not image_prompt_pairs:
            return []
        self._load()
        tok = self._processor.tokenizer

        def _first_token_ids(strings):
            ids = set()
            for s in strings:
                encoded = tok.encode(s, add_special_tokens=False)
                if encoded:
                    ids.add(encoded[0])
            return list(ids) or None

        yes_ids = _first_token_ids(["yes", "Yes", "YES"])
        no_ids  = _first_token_ids(["no",  "No",  "NO"])
        unc_ids = _first_token_ids(["uncertain", "Uncertain", "UNCERTAIN",
                                     "unsure", "Unsure"])

        def _best_logit(logits_row, ids):
            if not ids:
                return -1e9
            return max(logits_row[i].item() for i in ids)

        all_results = []
        for start in range(0, len(image_prompt_pairs), chunk_size):
            chunk = image_prompt_pairs[start: start + chunk_size]
            inputs = self._build_batch_inputs(chunk)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            last_logits = outputs.logits[:, -1, :]  # (chunk, vocab)

            for logits_row in last_logits:
                yes_l = _best_logit(logits_row, yes_ids)
                no_l  = _best_logit(logits_row, no_ids)
                unc_l = _best_logit(logits_row, unc_ids)
                max_l = max(yes_l, no_l, unc_l)
                e_yes = math.exp(yes_l - max_l)
                e_no  = math.exp(no_l  - max_l)
                e_unc = math.exp(unc_l - max_l)
                total = e_yes + e_no + e_unc
                p_yes = e_yes / total
                p_no  = e_no  / total
                p_unc = e_unc / total
                top2 = sorted([p_yes, p_no, p_unc], reverse=True)
                all_results.append({
                    "yes": p_yes, "no": p_no, "uncertain": p_unc,
                    "margin": top2[0] - top2[1],
                })
        return all_results

    def _get_yesno_probs_batch(self, image_question_pairs: list, chunk_size: int = 32) -> list:
        """
        Batched forward pass: extract P(yes), P(no), P(uncertain) for each (image, question) pair.
        Appends a bare yes/no/uncertain instruction suffix to each question.
        Use _forward_probs_batch directly when prompts are already fully formatted.
        """
        formatted = [
            (img, f"{q}\nAnswer with only 'yes', 'no', or 'uncertain'.")
            for img, q in image_question_pairs
        ]
        return self._forward_probs_batch(formatted, chunk_size=chunk_size)

    def _score_logit_gated_v2(self, image, item, questions: list, mode: str) -> dict:
        """Single-image logit scoring for gated_v2 modes. Called by score_image()."""
        n = len(questions)
        if n == 0:
            return {"score": 0.0, "question_scores": [], "component_scores": {},
                    "mode": mode, "vlm_seconds": 0.0, "num_questions": 0,
                    "reward_debug": {"reward_mode": mode}}
        t0 = time.time()
        probs_list = self._get_yesno_probs_batch(
            [(image, q.question) for q in questions]
        )
        vlm_seconds = time.time() - t0

        type_scores: dict = {}
        type_counts: dict = {}
        total_p_unc = 0.0
        question_scores = []
        for probs, q in zip(probs_list, questions):
            expected = q.answer.lower()
            q_type = getattr(q, "q_type", "unknown")
            q_score = (probs["yes"] + 0.5 * probs["uncertain"]) if expected == "yes" \
                      else (probs["no"] + 0.5 * probs["uncertain"])
            type_scores[q_type] = type_scores.get(q_type, 0.0) + q_score
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            total_p_unc += probs["uncertain"]
            predicted = max(["yes", "no", "uncertain"], key=lambda k: probs[k])
            question_scores.append({
                "question":    q.question,
                "expected":    expected,
                "predicted":   predicted,
                "correct":     predicted == expected,
                "score":       q_score,
                "weight":      1.0,
                "q_type":      q_type,
                "p_yes":       probs["yes"],
                "p_no":        probs["no"],
                "p_uncertain": probs["uncertain"],
                "margin":      probs["margin"],
            })

        comp = {qt: type_scores[qt] / type_counts[qt] for qt in type_scores}
        comp["uncertain_frac"] = total_p_unc / max(n, 1)
        bucket = getattr(item, "bucket", "")
        reward, debug = apply_gated_v2(mode, comp, bucket=bucket)

        self._total_vlm_calls += len(questions)
        self._total_questions_answered += n
        self._total_vlm_seconds += vlm_seconds
        return {
            "score":            float(reward),
            "question_scores":  question_scores,
            "component_scores": comp,
            "mode":             mode,
            "vlm_seconds":      vlm_seconds,
            "num_questions":    n,
            "reward_debug":     {**debug, "reward_mode": mode},
        }

    def _score_images_batch_logit_gated_v2(self, images_and_items: list, mode: str) -> list:
        """Batched logit scoring for gated_v2 modes. Called by score_images_batch()."""
        def _get_questions(item):
            qs = getattr(item, "grpo_reward_questions", []) or []
            return qs if qs else item.target_questions

        questions_list = [_get_questions(item) for _, item in images_and_items]

        # Flatten all (image, question) pairs with back-pointers
        flat_pairs: list = []
        flat_meta: list = []
        for i, ((img, _), qs) in enumerate(zip(images_and_items, questions_list)):
            for j, q in enumerate(qs):
                flat_pairs.append((img, q.question))
                flat_meta.append((i, j))

        t0 = time.time()
        flat_probs = self._get_yesno_probs_batch(flat_pairs)
        vlm_seconds = time.time() - t0
        per_img_s = vlm_seconds / max(len(images_and_items), 1)

        # Reassemble per-item probability lists
        item_probs: list = [[] for _ in images_and_items]
        for (i, _j), probs in zip(flat_meta, flat_probs):
            item_probs[i].append(probs)

        self._total_vlm_calls += len(flat_pairs)
        self._total_questions_answered += len(flat_pairs)
        self._total_vlm_seconds += vlm_seconds

        results = []
        for (_, item), qs, probs_list in zip(images_and_items, questions_list, item_probs):
            n = len(qs)
            type_scores: dict = {}
            type_counts: dict = {}
            total_p_unc = 0.0
            question_scores = []
            for probs, q in zip(probs_list, qs):
                expected = q.answer.lower()
                q_type = getattr(q, "q_type", "unknown")
                q_score = (probs["yes"] + 0.5 * probs["uncertain"]) if expected == "yes" \
                          else (probs["no"] + 0.5 * probs["uncertain"])
                type_scores[q_type] = type_scores.get(q_type, 0.0) + q_score
                type_counts[q_type] = type_counts.get(q_type, 0) + 1
                total_p_unc += probs["uncertain"]
                predicted = max(["yes", "no", "uncertain"], key=lambda k: probs[k])
                question_scores.append({
                    "question":    q.question,
                    "expected":    expected,
                    "predicted":   predicted,
                    "correct":     predicted == expected,
                    "score":       q_score,
                    "weight":      1.0,
                    "q_type":      q_type,
                    "p_yes":       probs["yes"],
                    "p_no":        probs["no"],
                    "p_uncertain": probs["uncertain"],
                    "margin":      probs["margin"],
                })
            comp = {qt: type_scores[qt] / type_counts[qt] for qt in type_scores}
            comp["uncertain_frac"] = total_p_unc / max(n, 1)
            bucket = getattr(item, "bucket", "")
            reward, debug = apply_gated_v2(mode, comp, bucket=bucket)
            results.append({
                "score":            float(reward),
                "question_scores":  question_scores,
                "component_scores": comp,
                "mode":             mode,
                "vlm_seconds":      per_img_s,
                "num_questions":    n,
                "reward_debug":     {**debug, "reward_mode": mode},
            })
        return results

    def _generate_with_scores(self, inputs, max_new_tokens: int):
        """Generate and return (text, scores) for first-token probability extraction."""
        import torch
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
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

    def _score_images_batch_contrastive_rubric(
        self, images_and_items: list, mode: str, tau: float = 0.20
    ) -> list:
        """
        Rubric-prompted contrastive scoring for grpo_attr_contrastive_rubric_v1
        and grpo_spatial_contrastive_rubric_v1.

        Parses obj/attr (or obj/relation) metadata from item questions, then builds
        COMBINED binding/relation statements for each image rather than scoring
        individual per-attribute questions independently. Batches all statements
        across all images in a single forward pass.

        For attribute:
          - presence, target_binding, swapped_binding, attr1, attr2, alignment, quality
          - contrastive_attr = sigmoid((target_binding_soft - swapped_binding_soft) / tau)

        For spatial:
          - presence, target_relation, opposite_relation, separation, alignment, quality
          - contrastive_relation = sigmoid((target_relation_soft - opposite_relation_soft) / tau)

        Falls back to individual question-based approach if metadata parsing fails.
        """
        def _get_questions(item):
            qs = getattr(item, "grpo_reward_questions", []) or []
            return qs if qs else item.target_questions

        def _sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x / tau))

        def _orig_prompt(item):
            return (getattr(item, "prompt", None) or getattr(item, "text", "") or "")

        # ── Build per-item statement dicts ────────────────────────────────────
        # Each entry: (ordered_labels: list[str], stmts: dict[label → text], parse_mode: str)
        item_build: list = []

        for img, item in images_and_items:
            questions = _get_questions(item)
            orig      = _orig_prompt(item)

            if mode == "grpo_attr_contrastive_rubric_v1":
                attr_pairs = _parse_attr_pairs_from_questions(questions)
                if len(attr_pairs) >= 2:
                    obj1, attr1 = attr_pairs[0]
                    obj2, attr2 = attr_pairs[1]
                    stmts = {
                        "presence":        f"Both requested objects are visible and recognizable in the image: a {obj1} and a {obj2}.",
                        "target_binding":  f"The image contains a visible {attr1} {obj1} and a visible {attr2} {obj2}, with the attributes bound to the correct objects.",
                        "swapped_binding": f"The image instead shows a visible {attr2} {obj1} and a visible {attr1} {obj2}.",
                        "attr1":           f"The {obj1} is visible and recognizable, and the {obj1} is {attr1}.",
                        "attr2":           f"The {obj2} is visible and recognizable, and the {obj2} is {attr2}.",
                        "alignment":       "The image broadly depicts the original text prompt without changing the main requested scene.",
                        "quality":         "The image is visually coherent enough to identify the main requested objects, without severe distortion or unreadable blobs.",
                    }
                    item_build.append((list(stmts.keys()), stmts, "attr_combined", questions))
                else:
                    # Fallback: score existing question texts with rubric wrapper
                    stmts = {f"q_{i}": q.question for i, q in enumerate(questions)}
                    item_build.append((list(stmts.keys()), stmts, "attr_fallback", questions))

            elif mode == "grpo_spatial_contrastive_rubric_v1":
                info = _parse_spatial_info_from_questions(questions)
                if info:
                    opp = info["opposite_relation"] or f"not {info['relation']}"
                    stmts = {
                        "presence":          f"Both requested objects are visible and recognizable in the image: {info['obj1']} and {info['obj2']}.",
                        "target_relation":   f"The {info['obj1']} is clearly {info['relation']} the {info['obj2']}.",
                        "opposite_relation": f"The {info['obj1']} is clearly {opp} the {info['obj2']}.",
                        "separation":        "The two requested objects are spatially separated enough that their relative positions can be judged.",
                        "alignment":         "The image broadly depicts the original text prompt without changing the main requested scene.",
                        "quality":           "The image is visually coherent enough to identify the main requested objects, without severe distortion or unreadable blobs.",
                    }
                    item_build.append((list(stmts.keys()), stmts, "spatial_combined", questions))
                else:
                    stmts = {f"q_{i}": q.question for i, q in enumerate(questions)}
                    item_build.append((list(stmts.keys()), stmts, "spatial_fallback", questions))

        # ── Flatten all (image, rubric_text) pairs ────────────────────────────
        flat_pairs: list = []
        flat_meta:  list = []   # (item_idx, label)
        for i, ((img, item), (labels, stmts, _, _qs)) in enumerate(
            zip(images_and_items, item_build)
        ):
            orig = _orig_prompt(item)
            for label in labels:
                rubric = RUBRIC_TEMPLATE.format(prompt=orig, statement=stmts[label])
                flat_pairs.append((img, rubric))
                flat_meta.append((i, label))

        t0 = time.time()
        flat_probs = self._forward_probs_batch(flat_pairs)
        vlm_seconds = time.time() - t0
        per_img_s   = vlm_seconds / max(len(images_and_items), 1)

        self._total_vlm_calls          += len(flat_pairs)
        self._total_questions_answered += len(flat_pairs)
        self._total_vlm_seconds        += vlm_seconds

        # Reassemble per-item: label → probs dict
        item_probs: list = [{} for _ in images_and_items]
        for (i, label), probs in zip(flat_meta, flat_probs):
            item_probs[i][label] = probs

        # ── Score each item ───────────────────────────────────────────────────
        results = []
        for i, ((_, item), (labels, stmts, parse_mode, questions)) in enumerate(
            zip(images_and_items, item_build)
        ):
            probs_map = item_probs[i]
            bucket    = getattr(item, "bucket", "")

            def _soft(label):
                p = probs_map.get(label, {"yes": 0.5, "no": 0.5, "uncertain": 0.0, "margin": 0.0})
                return p["yes"] + 0.5 * p["uncertain"]

            mean_margin    = sum(p.get("margin", 0.0) for p in probs_map.values()) / max(len(probs_map), 1)
            frac_uncertain = sum(
                1 for p in probs_map.values()
                if max(["yes", "no", "uncertain"], key=lambda k: p[k]) == "uncertain"
            ) / max(len(probs_map), 1)

            # ── Combined-statement path ───────────────────────────────────────
            if parse_mode == "attr_combined":
                presence        = _soft("presence")
                target_binding  = _soft("target_binding")
                swapped_binding = _soft("swapped_binding")
                attr1_score     = _soft("attr1")
                attr2_score     = _soft("attr2")
                alignment       = _soft("alignment")
                quality         = _soft("quality")
                attr_sub        = 0.5 * attr1_score + 0.5 * attr2_score
                contrastive     = _sigmoid(target_binding - swapped_binding)
                comp = {
                    "object_presence":    presence,
                    "target_binding":     target_binding,
                    "swapped_binding":    swapped_binding,
                    "contrastive_attr":   contrastive,
                    "attribute_subscore": attr_sub,
                    "attr1_score":        attr1_score,
                    "attr2_score":        attr2_score,
                    "prompt_alignment":   alignment,
                    "image_quality":      quality,
                    "uncertain_frac":     frac_uncertain,
                    "mean_logit_margin":  mean_margin,
                }

            elif parse_mode == "spatial_combined":
                presence     = _soft("presence")
                target_rel   = _soft("target_relation")
                opposite_rel = _soft("opposite_relation")
                separation   = _soft("separation")
                alignment    = _soft("alignment")
                quality      = _soft("quality")
                contrastive  = _sigmoid(target_rel - opposite_rel)
                comp = {
                    "object_presence":      presence,
                    "target_relation":      target_rel,
                    "opposite_relation":    opposite_rel,
                    "contrastive_relation": contrastive,
                    "separation_clarity":   separation,
                    "prompt_alignment":     alignment,
                    "image_quality":        quality,
                    "uncertain_frac":       frac_uncertain,
                    "mean_logit_margin":    mean_margin,
                }

            else:
                # ── Fallback: build comp from per-question scores ─────────────
                by_type_idx: dict = {}   # qt → list of (q, qi)
                for qi, q in enumerate(questions):
                    qt = getattr(q, "q_type", "unknown")
                    by_type_idx.setdefault(qt, []).append((q, qi))

                type_sums:   dict = {}
                type_counts: dict = {}
                total_p_unc       = 0.0
                for qi, q in enumerate(questions):
                    label = f"q_{qi}"
                    p     = probs_map.get(label, {"yes": 0.5, "no": 0.5, "uncertain": 0.0})
                    qt    = getattr(q, "q_type", "unknown")
                    exp   = q.answer.lower()
                    q_s   = (p["yes"] + 0.5 * p["uncertain"]) if exp == "yes" \
                            else (p["no"] + 0.5 * p["uncertain"])
                    total_p_unc         += p["uncertain"]
                    type_sums[qt]        = type_sums.get(qt, 0.0) + q_s
                    type_counts[qt]      = type_counts.get(qt, 0) + 1

                comp = {qt: type_sums[qt] / type_counts[qt] for qt in type_sums}
                comp["uncertain_frac"]   = total_p_unc / max(len(questions), 1)
                comp["mean_logit_margin"] = mean_margin

                # Best-effort contrastive from q_types
                if mode == "grpo_attr_contrastive_rubric_v1":
                    attr_qs = by_type_idx.get("attribute", [])
                    swap_qs = by_type_idx.get("anti_swap",  [])
                    pairs   = list(zip(attr_qs, swap_qs))
                    if pairs:
                        cs = [_sigmoid(_soft(f"q_{ia}") - _soft(f"q_{iswap}"))
                              for (_, ia), (_, iswap) in pairs]
                        comp["contrastive_attr"] = sum(cs) / len(cs)
                    else:
                        comp["contrastive_attr"] = comp.get("attribute", 0.5)
                elif mode == "grpo_spatial_contrastive_rubric_v1":
                    rel_qs  = by_type_idx.get("relation",      [])
                    anti_qs = by_type_idx.get("anti_relation",  [])
                    pairs   = list(zip(rel_qs, anti_qs))
                    if pairs:
                        cs = [_sigmoid(_soft(f"q_{ir}") - _soft(f"q_{ia}"))
                              for (_, ir), (_, ia) in pairs]
                        comp["contrastive_relation"] = sum(cs) / len(cs)
                    else:
                        comp["contrastive_relation"] = comp.get("relation", 0.5)

            reward, debug = apply_contrastive_rubric(mode, comp, bucket=bucket)

            # Build question_scores for logging (one entry per scored statement)
            question_scores = []
            for label in labels:
                p = probs_map.get(label, {"yes": 0.5, "no": 0.5, "uncertain": 0.0, "margin": 0.0})
                predicted = max(["yes", "no", "uncertain"], key=lambda k: p[k])
                question_scores.append({
                    "question":    stmts.get(label, label),
                    "expected":    "yes",
                    "predicted":   predicted,
                    "correct":     predicted == "yes",
                    "score":       p["yes"] + 0.5 * p["uncertain"],
                    "weight":      1.0,
                    "q_type":      label,
                    "p_yes":       p["yes"],
                    "p_no":        p["no"],
                    "p_uncertain": p["uncertain"],
                    "margin":      p.get("margin", 0.0),
                })

            results.append({
                "score":            float(reward),
                "question_scores":  question_scores,
                "component_scores": comp,
                "mode":             mode,
                "vlm_seconds":      per_img_s,
                "num_questions":    len(labels),
                "reward_debug":     {**debug, "reward_mode": mode},
            })
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

        # Contrastive rubric: rubric-prompted logit scoring with sigmoid contrastive pairs
        if mode in CONTRASTIVE_RUBRIC_MODES:
            return self._score_images_batch_contrastive_rubric(images_and_items, mode)

        # Gated-v2: batched logit scoring (numerically stable, no JSON generation)
        if mode in GATED_V2_MODES:
            return self._score_images_batch_logit_gated_v2(images_and_items, mode)

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
                    "qwen_logit_grpo_target_heavy") or mode in GATED_V2_MODES \
                    or mode in CONTRASTIVE_RUBRIC_MODES:
            questions = getattr(item, "grpo_reward_questions", []) or []
            if not questions:
                questions = item.target_questions
        else:
            questions = item.target_questions

        if mode in GATED_V2_MODES:
            return self._score_logit_gated_v2(image, item, questions, mode)

        if mode in CONTRASTIVE_RUBRIC_MODES:
            results = self._score_images_batch_contrastive_rubric([(image, item)], mode)
            return results[0]

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
