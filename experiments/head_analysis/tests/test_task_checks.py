import pytest

from experiments.head_analysis.analysis.task_checks import TaskCheckError, evaluate_ruler_answer


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        return "The answer is Needle Value."


def test_ruler_answer_check_normalizes_text():
    result = evaluate_ruler_answer(
        {"sample_id": "ruler", "reference_answer": ["needle   value"]}, [1, 2], FakeTokenizer()
    )

    assert result["answer_match"] is True


def test_ruler_answer_check_requires_one_reference():
    with pytest.raises(TaskCheckError, match="exactly one"):
        evaluate_ruler_answer({"sample_id": "ruler", "reference_answer": []}, [1], FakeTokenizer())
