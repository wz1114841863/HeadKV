"""Task-specific, read-only checks for completed B1 generated-ID artifacts."""

from __future__ import annotations

from typing import Any, Sequence


class TaskCheckError(RuntimeError):
    """Raised when a frozen task contract cannot be evaluated."""


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def evaluate_ruler_answer(record: dict[str, Any], generated_ids: Sequence[int], tokenizer) -> dict[str, Any]:
    """Check whether one frozen RULER needle answer occurs in generated text."""

    sample_id = record.get("sample_id")
    answers = record.get("reference_answer")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
        raise TaskCheckError(f"{sample_id}: expected exactly one text reference_answer")
    generated_text = tokenizer.decode(list(generated_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not isinstance(generated_text, str):
        raise TaskCheckError(f"{sample_id}: tokenizer.decode did not return text")
    answer = answers[0]
    return {
        "sample_id": sample_id,
        "reference_answer": answer,
        "answer_match": normalize_text(answer) in normalize_text(generated_text),
        "generated_text": generated_text,
    }
