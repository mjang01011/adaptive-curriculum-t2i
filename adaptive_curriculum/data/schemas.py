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
    target_questions: List[EvalQuestion]
    grpo_reward_questions: List[EvalQuestion] = field(default_factory=list)
    reward_rule: Dict[str, Any] = field(default_factory=dict)
    grpo_reward_rule: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None

    @property
    def text(self) -> str:
        return self.prompt

    @classmethod
    def from_dict(cls, d: dict) -> "BucketItem":
        raw_qs = d.get("target_questions") or d.get("eval_questions") or []
        questions = [
            EvalQuestion(
                question=q["question"],
                answer=q["answer"],
                q_type=q.get("type", q.get("q_type", "")),
                weight=q.get("weight", 1.0),
            )
            for q in raw_qs
        ]
        raw_grpo_qs = d.get("grpo_reward_questions") or []
        grpo_questions = [
            EvalQuestion(
                question=q["question"],
                answer=q["answer"],
                q_type=q.get("type", q.get("q_type", "")),
                weight=q.get("weight", 1.0),
            )
            for q in raw_grpo_qs
        ]
        return cls(
            id=d["id"],
            bucket=d["bucket"],
            prompt=d.get("prompt") or d.get("caption") or "",
            target_questions=questions,
            grpo_reward_questions=grpo_questions,
            reward_rule=d.get("reward_rule") or d.get("target_reward_rule", {}),
            grpo_reward_rule=d.get("grpo_reward_rule", {}),
            image_path=d.get("image_path"),
        )

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "id": self.id,
            "bucket": self.bucket,
            "prompt": self.prompt,
            "target_questions": [
                {"type": q.q_type, "question": q.question, "answer": q.answer}
                for q in self.target_questions
            ],
            "grpo_reward_questions": [
                {"type": q.q_type, "question": q.question, "answer": q.answer, "weight": q.weight}
                for q in self.grpo_reward_questions
            ],
            "reward_rule": self.reward_rule,
            "grpo_reward_rule": self.grpo_reward_rule,
        }
        if self.image_path is not None:
            d["image_path"] = self.image_path
        return d
