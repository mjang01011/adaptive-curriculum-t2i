"""
CARGORewardModel: extends Qwen3VLRewardModel with CARGO v2 reward modes.

New in v2 vs v1:
  grpo_attr_contrastive_rubric_v2
      Same statement set as v1; dispatches to v2 reward formula (higher
      attribute_sub weight, explicit target_binding term).

  grpo_spatial_contrastive_rubric_v2
      Adds a "distinct_objects" rubric statement to score whether the image
      shows two separately identifiable objects (not a patch/shadow).
      Dispatches to v2 reward formula (triple gate: contrastive * presence * distinct).

All non-CARGO modes are delegated to the parent Qwen3VLRewardModel.
"""
import math
import time

from adaptive_curriculum.reward.vlm_reward import (
    Qwen3VLRewardModel,
    RUBRIC_TEMPLATE,
    _parse_attr_pairs_from_questions,
    _parse_spatial_info_from_questions,
)
from CARGO.rewards import CARGO_MODES, apply_cargo_reward


class CARGORewardModel(Qwen3VLRewardModel):
    """
    Extends Qwen3VLRewardModel to support CARGO v2 scoring modes.
    All non-CARGO modes fall through to the parent implementation.
    """

    def _score_images_batch_cargo_v2(
        self,
        images_and_items: list,
        mode: str,
        tau: float = 0.20,
    ) -> list:
        """
        Rubric-prompted batch scoring for CARGO v2 modes.

        Attribute v2: identical statements to v1; different reward formula.
        Spatial v2:   adds "distinct_objects" statement for the triple gate.

        All statements for all images are batched into a single _forward_probs_batch
        call for maximum throughput.
        """
        def _get_questions(item):
            qs = getattr(item, "grpo_reward_questions", []) or []
            return qs if qs else item.target_questions

        def _sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x / tau))

        def _orig_prompt(item):
            return getattr(item, "prompt", None) or getattr(item, "text", "") or ""

        # ── Build statement dicts for each image ─────────────────────────────
        item_build = []
        for img, item in images_and_items:
            questions = _get_questions(item)

            if mode == "grpo_attr_contrastive_rubric_v2":
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
                    item_build.append((list(stmts.keys()), stmts, "attr_combined_v2", questions))
                else:
                    stmts = {f"q_{i}": q.question for i, q in enumerate(questions)}
                    item_build.append((list(stmts.keys()), stmts, "attr_fallback", questions))

            elif mode == "grpo_spatial_contrastive_rubric_v2":
                info = _parse_spatial_info_from_questions(questions)
                if info:
                    opp = info["opposite_relation"] or f"not {info['relation']}"
                    stmts = {
                        "presence":          f"Both requested objects are visible and recognizable in the image: {info['obj1']} and {info['obj2']}.",
                        "distinct_objects":  (
                            f"There are two distinct, separately identifiable objects in the image: "
                            f"one {info['obj1']} and one {info['obj2']}. They are not merely two "
                            f"colored regions of the same object, an attached patch, a shadow, or "
                            f"an ambiguous fused shape."
                        ),
                        "target_relation":   f"The {info['obj1']} is clearly {info['relation']} the {info['obj2']}.",
                        "opposite_relation": f"The {info['obj1']} is clearly {opp} the {info['obj2']}.",
                        "separation":        "The two requested objects are spatially separated enough that their relative positions can be judged.",
                        "alignment":         "The image broadly depicts the original text prompt without changing the main requested scene.",
                        "quality":           "The image is visually coherent enough to identify the main requested objects, without severe distortion or unreadable blobs.",
                    }
                    item_build.append((list(stmts.keys()), stmts, "spatial_combined_v2", questions))
                else:
                    stmts = {f"q_{i}": q.question for i, q in enumerate(questions)}
                    item_build.append((list(stmts.keys()), stmts, "spatial_fallback", questions))

        # ── Flatten all (image, rubric_text) pairs ────────────────────────────
        flat_pairs = []
        flat_meta  = []
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
        item_probs = [{} for _ in images_and_items]
        for (i, label), probs in zip(flat_meta, flat_probs):
            item_probs[i][label] = probs

        # ── Score each item ───────────────────────────────────────────────────
        results = []
        for i, ((_, item), (labels, stmts, parse_mode, questions)) in enumerate(
            zip(images_and_items, item_build)
        ):
            probs_map = item_probs[i]
            bucket    = getattr(item, "bucket", "")

            def _soft(label, _pm=probs_map):
                p = _pm.get(label, {"yes": 0.5, "no": 0.5, "uncertain": 0.0, "margin": 0.0})
                return p["yes"] + 0.5 * p["uncertain"]

            mean_margin    = (
                sum(p.get("margin", 0.0) for p in probs_map.values())
                / max(len(probs_map), 1)
            )
            frac_uncertain = (
                sum(1 for p in probs_map.values()
                    if max(["yes", "no", "uncertain"], key=lambda k: p[k]) == "uncertain")
                / max(len(probs_map), 1)
            )

            if parse_mode == "attr_combined_v2":
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

            elif parse_mode == "spatial_combined_v2":
                presence     = _soft("presence")
                distinct     = _soft("distinct_objects")
                target_rel   = _soft("target_relation")
                opposite_rel = _soft("opposite_relation")
                separation   = _soft("separation")
                alignment    = _soft("alignment")
                quality      = _soft("quality")
                contrastive  = _sigmoid(target_rel - opposite_rel)
                comp = {
                    "object_presence":       presence,
                    "distinct_object_score": distinct,
                    "target_relation":       target_rel,
                    "opposite_relation":     opposite_rel,
                    "contrastive_relation":  contrastive,
                    "separation_clarity":    separation,
                    "prompt_alignment":      alignment,
                    "image_quality":         quality,
                    "uncertain_frac":        frac_uncertain,
                    "mean_logit_margin":     mean_margin,
                }

            else:
                # Fallback: per-question soft scoring from existing q_types
                by_type_idx = {}
                for qi, q in enumerate(questions):
                    qt = getattr(q, "q_type", "unknown")
                    by_type_idx.setdefault(qt, []).append((q, qi))

                type_sums   = {}
                type_counts = {}
                total_p_unc = 0.0
                for qi, q in enumerate(questions):
                    label_q = f"q_{qi}"
                    p     = probs_map.get(label_q, {"yes": 0.5, "no": 0.5, "uncertain": 0.0})
                    qt    = getattr(q, "q_type", "unknown")
                    exp   = q.answer.lower()
                    q_s   = (p["yes"] + 0.5 * p["uncertain"]) if exp == "yes" \
                            else (p["no"] + 0.5 * p["uncertain"])
                    total_p_unc         += p["uncertain"]
                    type_sums[qt]        = type_sums.get(qt, 0.0) + q_s
                    type_counts[qt]      = type_counts.get(qt, 0) + 1

                comp = {qt: type_sums[qt] / type_counts[qt] for qt in type_sums}
                comp["uncertain_frac"]    = total_p_unc / max(len(questions), 1)
                comp["mean_logit_margin"] = mean_margin

                if mode == "grpo_attr_contrastive_rubric_v2":
                    attr_qs = by_type_idx.get("attribute", [])
                    swap_qs = by_type_idx.get("anti_swap",  [])
                    pairs   = list(zip(attr_qs, swap_qs))
                    if pairs:
                        cs = [_sigmoid(_soft(f"q_{ia}") - _soft(f"q_{is_}"))
                              for (_, ia), (_, is_) in pairs]
                        comp["contrastive_attr"] = sum(cs) / len(cs)
                    else:
                        comp["contrastive_attr"] = comp.get("attribute", 0.5)
                    comp.setdefault("target_binding", comp.get("attribute", 0.5))
                    comp.setdefault("attribute_subscore", comp.get("attribute", 0.5))

                elif mode == "grpo_spatial_contrastive_rubric_v2":
                    rel_qs  = by_type_idx.get("relation",      [])
                    anti_qs = by_type_idx.get("anti_relation",  [])
                    pairs   = list(zip(rel_qs, anti_qs))
                    if pairs:
                        cs = [_sigmoid(_soft(f"q_{ir}") - _soft(f"q_{ia}"))
                              for (_, ir), (_, ia) in pairs]
                        comp["contrastive_relation"] = sum(cs) / len(cs)
                    else:
                        comp["contrastive_relation"] = comp.get("relation", 0.5)
                    comp.setdefault("distinct_object_score", 0.5)

            reward, debug = apply_cargo_reward(mode, comp, bucket=bucket)

            # Merge derived scores computed inside the reward formula back into
            # comp so that component_scores contains relation_effective (spatial)
            # and contrastive_attr (attribute), which are active CARGO components.
            for k, v in debug.items():
                if k not in ("uncertain_penalty", "reward_mode"):
                    comp.setdefault(k, v)

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
        if mode in CARGO_MODES:
            return self._score_images_batch_cargo_v2(images_and_items, mode)
        return super().score_images_batch(images_and_items, mode)

    def score_image(self, image, item, mode: str = "hard_target") -> dict:
        if mode in CARGO_MODES:
            return self._score_images_batch_cargo_v2([(image, item)], mode)[0]
        return super().score_image(image, item, mode=mode)
