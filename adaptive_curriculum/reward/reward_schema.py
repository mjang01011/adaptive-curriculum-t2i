from typing import List, Union


class RewardModel:
    def score_image(self, image, item, mode: str = "hard_target") -> dict:
        """
        image: file path (str) or PIL.Image.Image (in-memory, no disk I/O).
        mode: "hard_target" (uncertain=0), "pseudo_soft" (uncertain=0.5), "true_soft" (VLM logits).
        Returns {"score": float, "question_scores": [...], "mode": mode}
        """
        raise NotImplementedError

    def score_batch(self, images: List[Union[str, object]], items: list, mode: str = "hard_target") -> List[dict]:
        return [self.score_image(img, item, mode=mode) for img, item in zip(images, items)]
