from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EvalQuestion:
    question: str
    answer: str          # "yes" | "no"
    q_type: str = ""     # "attribute", "anti_swap", "count", "relation", etc.
    weight: float = 1.0


@dataclass
class BucketItem:
    id: str
    bucket: str
    prompt: str
    eval_questions: List[EvalQuestion]
    reward_rule: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None

    @property
    def text(self) -> str:
        return self.prompt

    @classmethod
    def from_dict(cls, d: dict) -> "BucketItem":
        # support both "target_questions" (actual JSONL) and "eval_questions" (legacy)
        raw_qs = d.get("target_questions") or d.get("eval_questions") or []
        questions = [
            EvalQuestion(
                question=q["question"],
                answer=q["answer"],
                q_type=q.get("type", ""),
                weight=q.get("weight", 1.0),
            )
            for q in raw_qs
        ]
        return cls(
            id=d["id"],
            bucket=d["bucket"],
            prompt=d.get("prompt") or d.get("caption") or "",
            eval_questions=questions,
            reward_rule=d.get("reward_rule", {}),
            image_path=d.get("image_path"),
        )

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "id": self.id,
            "bucket": self.bucket,
            "prompt": self.prompt,
            "target_questions": [
                {"type": q.q_type, "question": q.question, "answer": q.answer}
                for q in self.eval_questions
            ],
            "reward_rule": self.reward_rule,
        }
        if self.image_path is not None:
            d["image_path"] = self.image_path
        return d
