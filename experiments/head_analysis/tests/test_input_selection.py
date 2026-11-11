import json

import pytest

from experiments.head_analysis.runners.input_selection import B1RunnerError, load_frozen_selection


def test_selection_preserves_requested_order_and_input_ids(tmp_path):
    records = [
        {"sample_id": name, "input_ids": [1, index], "actual_prompt_length": 2, "max_new_tokens": 8}
        for index, name in enumerate(("first", "second"))
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    selected, _ = load_frozen_selection(path, manifest, ["second", "first"])
    assert selected == records[::-1]


@pytest.mark.parametrize("requested", [["first", "first"], ["missing"], []])
def test_selection_rejects_invalid_requests(tmp_path, requested):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"sample_id": "first", "input_ids": [1], "input_length": 1, "max_new_tokens": 2}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(B1RunnerError):
        load_frozen_selection(path, manifest, requested)


def test_selection_rejects_invalid_token_length(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"sample_id": "first", "input_ids": [1], "input_length": 2, "max_new_tokens": 2}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(B1RunnerError, match="length differs"):
        load_frozen_selection(path, manifest, ["first"])
