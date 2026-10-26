import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "prompt_decode_smoke.py"
SPEC = importlib.util.spec_from_file_location("prompt_decode_smoke", SCRIPT)
prompt_decode_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prompt_decode_smoke)


class TinyTokenizer:
    def __len__(self):
        return 512

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}<assistant>"

    def encode(self, text, add_special_tokens):
        return [ord(char) for char in text]


def make_record(dataset, target, index):
    tokenizer = TinyTokenizer()
    content = f"{dataset}-{target}-{index}"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], False, True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {
        "sample_id": f"{dataset}_{target}_{index}",
        "dataset": dataset,
        "task": "smoke",
        "target_prompt_length": target,
        "actual_prompt_length": len(ids),
        "max_new_tokens": 16,
        "user_content": content,
        "assistant_prefix": "",
        "prompt": prompt,
        "input_ids": ids,
        "input_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "metadata": {},
    }


def test_select_smoke_records_has_two_per_required_bucket():
    records = [
        make_record(dataset, target, index)
        for dataset, target in prompt_decode_smoke.SMOKE_BUCKETS
        for index in range(3)
    ]
    selected = prompt_decode_smoke.select_smoke_records(records, per_bucket=2)
    assert len(selected) == 12
    assert {(record["dataset"], record["target_prompt_length"]) for record in selected} == set(
        prompt_decode_smoke.SMOKE_BUCKETS
    )


def test_validate_saved_record_detects_and_accepts_exact_ids():
    record = make_record("pg19", 8192, 1)
    passed = prompt_decode_smoke.validate_saved_record(TinyTokenizer(), record)
    assert passed["input_integrity_passed"]
    assert passed["input_length"] == len(record["input_ids"])

    record["input_ids"] = record["input_ids"][:-1]
    failed = prompt_decode_smoke.validate_saved_record(TinyTokenizer(), record)
    assert not failed["input_integrity_passed"]
    assert any("differs" in error or "does not match" in error for error in failed["input_integrity_errors"])


def test_generation_metrics_flags_short_pg19_and_ruler_answer():
    pg = make_record("pg19", 8192, 1)
    pg_metrics = prompt_decode_smoke.generation_metrics(
        pg,
        [1, 1, 1, 1],
        "repeated",
        requested_new_tokens=1024,
        eos_token_ids={2},
        pg19_min_new_tokens=512,
        repeat_ngram_threshold=0.6,
        unique_token_threshold=0.2,
        task_diagnostics=True,
    )
    assert "pg19_shorter_than_min_decode" in pg_metrics["anomalies"]

    ruler = make_record("ruler", 8192, 1)
    ruler["reference_answer"] = ["42"]
    ruler_metrics = prompt_decode_smoke.generation_metrics(
        ruler,
        [9, 10],
        "The answer is 42.",
        requested_new_tokens=128,
        eos_token_ids=set(),
        pg19_min_new_tokens=512,
        repeat_ngram_threshold=0.6,
        unique_token_threshold=0.2,
        task_diagnostics=True,
    )
    assert ruler_metrics["ruler_answer_match"] is True
