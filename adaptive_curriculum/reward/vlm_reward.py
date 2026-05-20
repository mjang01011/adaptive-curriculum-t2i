"""
VLM-based compositional reward.
Queries a vision-language model with yes/no questions about generated images.
"""
import base64
import os
from pathlib import Path
from typing import List, Optional

from adaptive_curriculum.reward.reward_schema import RewardModel


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMRewardModel(RewardModel):
    """
    Wraps an Anthropic Claude vision model for QA-based compositional scoring.
    Requires ANTHROPIC_API_KEY in environment.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 32):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def _ask_question(self, image_path: str, question: str, expected: str) -> dict:
        client = self._get_client()
        img_b64 = _image_to_base64(image_path)
        ext = Path(image_path).suffix.lstrip(".").lower()
        media_type = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "webp") else "image/png"

        prompt = (
            f"{question}\n"
            "Answer with only 'yes' or 'no'."
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        predicted = response.content[0].text.strip().lower()
        predicted = "yes" if "yes" in predicted else "no"
        correct = predicted == expected.lower()
        return {"predicted": predicted, "correct": correct, "raw_response": response.content[0].text}

    def score_image(self, image_path: str, item) -> dict:
        question_scores = []
        total_weight = sum(q.weight for q in item.eval_questions)
        weighted_sum = 0.0

        for q in item.eval_questions:
            result = self._ask_question(image_path, q.question, q.answer)
            weighted_sum += q.weight * float(result["correct"])
            question_scores.append({
                "question": q.question,
                "expected": q.answer,
                "predicted": result["predicted"],
                "correct": result["correct"],
                "weight": q.weight,
                "raw_response": result["raw_response"],
            })

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return {
            "score": float(score),
            "question_scores": question_scores,
            "raw_response": None,
        }


def build_reward_model(config) -> RewardModel:
    reward_type = getattr(config.evaluation, "reward_model", "heuristic")
    if reward_type == "heuristic":
        from adaptive_curriculum.reward.heuristic_reward import HeuristicRewardModel
        return HeuristicRewardModel()
    elif reward_type == "vlm":
        model_name = getattr(config.evaluation, "vlm_model", "claude-haiku-4-5-20251001")
        return VLMRewardModel(model=model_name)
    else:
        raise ValueError(f"Unknown reward_model: {reward_type}")
