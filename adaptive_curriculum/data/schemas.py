from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EvalQuestion:
    question: str
    answer: str
    weight: float = 1.0


@dataclass
class StructuredPrompt:
    objects: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    relations: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class BucketItem:
    id: str
    bucket: str
    eval_questions: List[EvalQuestion]
    # At least one of these must be present
    prompt: Optional[str] = None          # text-only validation item
    caption: Optional[str] = None        # caption when image_path is available
    image_path: Optional[str] = None     # path to training image (may be absent)
    structured_prompt: Optional[StructuredPrompt] = None

    @property
    def text(self) -> str:
        return self.caption or self.prompt or ""

    @classmethod
    def from_dict(cls, d: dict) -> "BucketItem":
        questions = [
            EvalQuestion(
                question=q["question"],
                answer=q["answer"],
                weight=q.get("weight", 1.0),
            )
            for q in d.get("eval_questions", [])
        ]
        sp = None
        if "structured_prompt" in d and d["structured_prompt"]:
            raw = d["structured_prompt"]
            sp = StructuredPrompt(
                objects=raw.get("objects", []),
                attributes=raw.get("attributes", {}),
                relations=raw.get("relations", []),
            )
        return cls(
            id=d["id"],
            bucket=d["bucket"],
            eval_questions=questions,
            prompt=d.get("prompt"),
            caption=d.get("caption"),
            image_path=d.get("image_path"),
            structured_prompt=sp,
        )

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "id": self.id,
            "bucket": self.bucket,
            "eval_questions": [
                {"question": q.question, "answer": q.answer, "weight": q.weight}
                for q in self.eval_questions
            ],
        }
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.caption is not None:
            d["caption"] = self.caption
        if self.image_path is not None:
            d["image_path"] = self.image_path
        if self.structured_prompt is not None:
            d["structured_prompt"] = {
                "objects": self.structured_prompt.objects,
                "attributes": self.structured_prompt.attributes,
                "relations": self.structured_prompt.relations,
            }
        return d
