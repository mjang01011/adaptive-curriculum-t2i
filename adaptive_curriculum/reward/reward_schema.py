from typing import List, Optional


class RewardModel:
    def score_image(self, image_path: str, item) -> dict:
        """
        Returns:
        {
          "score": float in [0, 1],
          "question_scores": [
            {
              "question": str,
              "expected": str,
              "predicted": str,
              "correct": bool,
              "weight": float,
            }
          ],
          "raw_response": optional str,
        }
        """
        raise NotImplementedError

    def score_batch(self, image_paths: List[str], items: list) -> List[dict]:
        return [self.score_image(p, item) for p, item in zip(image_paths, items)]
