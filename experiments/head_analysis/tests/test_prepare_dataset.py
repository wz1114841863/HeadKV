import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
prepare_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_dataset)


def test_parquet_loaders_and_pg19_metadata(tmp_path):
    gov_file = tmp_path / "gov.parquet"
    pq.write_table(
        pa.table({"report": ["report body"], "summary": ["summary body"]}),
        gov_file,
    )
    assert prepare_dataset.load_gov(str(gov_file)) == [{"report": "report body", "summary": "summary body"}]

    pg_dir = tmp_path / "pg19"
    pg_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "short_book_title": ["A short title"],
                "url": ["https://www.gutenberg.org/ebooks/123"],
                "text": ["book text"],
            }
        ),
        pg_dir / "test.parquet",
    )
    rows = prepare_dataset.load_pg19(str(pg_dir), "auto")
    assert rows[0]["text"] == "book text"
    assert prepare_dataset.pg_fields(rows[0], 0) == (
        "https://www.gutenberg.org/ebooks/123",
        "A short title",
        "book text",
    )


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def encode(self, text, add_special_tokens, verbose):
        return list(range(len(text.split())))


def test_gov_targets_are_disjoint_and_length_labeled():
    rows = [
        {"id": "a", "report": "one two", "summary": "s"},
        {"id": "b", "report": "one two three", "summary": "s"},
        {"id": "c", "report": "one two three four", "summary": "s"},
        {"id": "d", "report": "one two three four five", "summary": "s"},
    ]
    base_length = len(prepare_dataset.GOV_PREFIX.split()) + len(prepare_dataset.GOV_SUFFIX.split())
    first_target = base_length + 2
    second_target = base_length + 5
    records, report = prepare_dataset.select_gov_targets(
        TinyTokenizer(),
        rows,
        targets=[first_target, second_target],
        n_per_target=1,
        tolerance=0,
        seed=42,
        max_new_tokens=1,
    )
    assert [r["sample_id"] for r in records] == [
        f"govreport_{first_target}_001",
        f"govreport_{second_target}_001",
    ]
    assert len({r["metadata"]["report_sha256"] for r in records}) == 2
    assert report["targets"][str(first_target)]["selected"] == 1
    assert report["targets"][str(second_target)]["selected"] == 1
