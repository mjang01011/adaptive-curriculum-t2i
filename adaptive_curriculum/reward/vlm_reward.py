"""
VLM-based compositional reward using local Qwen3-VL.
"""
from pathlib import Path
from typing import List

from adaptive_curriculum.reward.reward_schema import RewardModel


class Qwen3VLRewardModel(RewardModel):
    """
    Runs Qwen3-VL-4B-Instruct locally. No API key needed.
    Answers yes/no/uncertain per target_question, scores via reward_rule formula.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

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

    def _ask(self, image_path: str, question: str) -> str:
        """Returns 'yes', 'no', or 'uncertain'."""
        import torch
        from qwen_vl_utils import process_vision_info

        self._load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Path(image_path).as_uri()},
                    {"type": "text", "text": f"{question}\nAnswer with only 'yes', 'no', or 'uncertain'."},
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

        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=8)
        raw = self._processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip().lower()

        if "uncertain" in raw:
            return "uncertain"
        if "yes" in raw:
            return "yes"
        return "no"

    def score_image(self, image_path: str, item) -> dict:
        reward_rule = getattr(item, "reward_rule", {}) or {}
        uncertain_score = reward_rule.get("uncertain_score", 0.0)

        question_scores = []
        correct_count = 0.0

        for q in item.eval_questions:
            predicted = self._ask(image_path, q.question)
            if predicted == "uncertain":
                correct = False
                q_score = uncertain_score
            else:
                correct = predicted == q.answer.lower()
                q_score = 1.0 if correct else 0.0
            correct_count += q_score
            question_scores.append({
                "question": q.question,
                "expected": q.answer,
                "predicted": predicted,
                "correct": correct,
                "q_type": q.q_type,
            })

        n = len(item.eval_questions)
        score = correct_count / n if n > 0 else 0.0
        return {
            "score": float(score),
            "question_scores": question_scores,
        }


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
