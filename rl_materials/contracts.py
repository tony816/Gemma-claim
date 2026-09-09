"""Structural checks only. These deliberately cannot certify a legal reward."""
from __future__ import annotations

import re
from typing import Any


def eligible_cards(cards: list[dict], jurisdiction: str,
                   blocked_patent_ids: set[str]) -> dict[str, dict]:
    """Conservative card selection. Unknown family links never become context.

    The caller supplies the target's full reviewed patent family. Equality of
    publication IDs alone is insufficient to establish family disjointness.
    An empty blocked set represents missing target-family review and abstains.
    """
    if jurisdiction not in {"KR", "US"} or not blocked_patent_ids:
        return {}
    result = {}
    for card in cards:
        if (card.get("jurisdiction") != jurisdiction
                or card.get("review_status") != "approved"
                or card.get("source_text_verified") is not True
                or card.get("generator_eligible") is not True
                or card.get("legal_history_status") != "reviewed"
                or card.get("case_family_status") != "reviewed"):
            continue
        family = card.get("case_patent_family_ids")
        if not isinstance(family, list) or not family or any(not isinstance(x, str) or not x for x in family):
            continue
        if blocked_patent_ids.intersection(family):
            continue
        card_id = card.get("card_id")
        if not isinstance(card_id, str) or not card_id:
            continue
        if card_id in result:
            raise ValueError(f"Duplicate approved card ID: {card_id}")
        result[card_id] = card
    return result


def validate_output(output: Any, supplied_cards: dict[str, dict],
                    max_dependents: int = 3) -> list[str]:
    errors: list[str] = []
    if not isinstance(output, dict):
        return ["output_not_object"]
    if set(output) != {"claims", "annotations", "abstentions"}:
        errors.append("output_keys")
    claims = output.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= max_dependents + 1:
        return errors + ["claim_count"]
    texts: set[str] = set()
    for number, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            errors.append(f"claim_{number}_not_object")
            continue
        if set(claim) != {"number", "kind", "depends_on", "text"}:
            errors.append(f"claim_{number}_keys")
        if type(claim.get("number")) is not int or claim["number"] != number:
            errors.append(f"claim_{number}_number")
        expected = "independent" if number == 1 else "dependent"
        if claim.get("kind") != expected:
            errors.append(f"claim_{number}_kind")
        parents = claim.get("depends_on")
        valid_parents = isinstance(parents, list) and (
            parents == [] if number == 1 else
            len(parents) == 1 and type(parents[0]) is int and 1 <= parents[0] < number
        )
        if not valid_parents:
            errors.append(f"claim_{number}_dependency")
        value = claim.get("text")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"claim_{number}_empty_text")
            continue
        normalized = " ".join(value.split()).casefold()
        if normalized in texts:
            errors.append(f"claim_{number}_duplicate_text")
        texts.add(normalized)
        if number > 1 and valid_parents:
            references = [int(number_text) for groups in re.findall(
                r"(?:제\s*(\d+)\s*항|claim\s+(\d+))", value, re.I
            ) for number_text in groups if number_text]
            if references != parents:
                errors.append(f"claim_{number}_text_dependency")
    annotations = output.get("annotations")
    if not isinstance(annotations, list):
        errors.append("annotations_not_list")
    else:
        seen = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                errors.append("annotation_not_object")
                continue
            if set(annotation) != {"claim_number", "card_id", "mode", "application"}:
                errors.append("annotation_keys")
            n, card_id = annotation.get("claim_number"), annotation.get("card_id")
            if type(n) is not int or not 1 <= n <= len(claims):
                errors.append("annotation_claim")
            card = supplied_cards.get(card_id) if isinstance(card_id, str) else None
            if not card or card.get("review_status") != "approved":
                errors.append("annotation_unapproved_or_unsupplied_card")
            if annotation.get("mode") != "provided_during_drafting":
                # Post-hoc annotations are produced in a separate review flow.
                errors.append("annotation_mode")
            if not isinstance(annotation.get("application"), str) or not annotation["application"].strip():
                errors.append("annotation_application")
            if type(n) is int and isinstance(card_id, str):
                key = (n, card_id)
                if key in seen:
                    errors.append("annotation_duplicate")
                seen.add(key)
    abstentions = output.get("abstentions")
    if not isinstance(abstentions, list) or any(not isinstance(x, str) or not x.strip() for x in abstentions):
        errors.append("abstentions_format")
    return errors


def resolve_annotations(output: dict, cards: dict[str, dict]) -> list[dict]:
    """Resolve immutable source metadata, never trust model-written case names."""
    errors = validate_output(output, cards)
    if errors:
        raise ValueError(",".join(errors))
    return [{**annotation, "source": {key: cards[annotation["card_id"]][key]
             for key in ("jurisdiction", "docket", "court", "decision_date",
                         "pdf_path", "source_sha256", "pages")}}
            for annotation in output["annotations"]]


def assess(output: Any, cards: dict[str, dict]) -> dict:
    errors = validate_output(output, cards)
    return {"structural_errors": errors, "structurally_valid": not errors,
            "reward": None, "training_eligible": False,
            "unresolved": ["drawing_grounding", "meaningful_dependent_limitations",
                           "semantic_citation_support", "legal_scope"]}
