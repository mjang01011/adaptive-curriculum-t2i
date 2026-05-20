from typing import List, Union


class RewardModel:
    def score_image(self, image, item) -> dict:
        """
        image: file path (str) or PIL.Image.Image (in-memory, no disk I/O).
        Returns {"score": float, "question_scores": [...]}
        """
        raise NotImplementedError

    def score_batch(self, images: List[Union[str, object]], items: list) -> List[dict]:
        return [self.score_image(img, item) for img, item in zip(images, items)]
