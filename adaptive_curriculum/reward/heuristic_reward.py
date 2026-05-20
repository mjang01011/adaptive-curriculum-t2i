"""
Deterministic heuristic reward — no VLM required.
Used for smoke-test runs and pipeline validation.
"""
import hashlib
from typing import List

from adaptive_curriculum.reward.reward_schema import RewardModel


def _fake_score(item_id: str, image_path: str) -> float:
    key = f"{item_id}|{image_path}"
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class HeuristicRewardModel(RewardModel):
    def score_image(self, image_path: str, item) -> dict:
        base_score = _fake_score(item.id, image_path)

        question_scores = []
        total_weight = sum(q.weight for q in item.eval_questions)
        weighted_sum = 0.0
        for q in item.eval_questions:
            q_key = f"{item.id}|{image_path}|{q.question}"
            q_h = hashlib.sha256(q_key.encode()).hexdigest()
            q_score = int(q_h[:8], 16) / 0xFFFFFFFF
            correct = q_score > 0.5
            predicted = q.answer if correct else ("no" if q.answer == "yes" else "yes")
            weighted_sum += q.weight * float(correct)
            question_scores.append({
                "question": q.question,
                "expected": q.answer,
                "predicted": predicted,
                "correct": correct,
                "weight": q.weight,
            })

        score = weighted_sum / total_weight if total_weight > 0 else base_score
        return {
            "score": float(score),
            "question_scores": question_scores,
            "raw_response": None,
        }

    def score_batch(self, image_paths: List[str], items: list) -> List[dict]:
        return [self.score_image(p, item) for p, item in zip(image_paths, items)]
