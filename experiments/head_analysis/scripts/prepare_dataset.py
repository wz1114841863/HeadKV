import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

GOV_PREFIX = (
    "Read the following government report carefully. Produce a detailed, "
    "structured summary covering the main findings, supporting evidence, "
    "important relationships, and conclusions. Do not invent facts.\n\nReport:\n"
)
GOV_SUFFIX = "\n\nDetailed summary:"
PG_PREFIX = (
    "Continue the following passage naturally. Preserve its style and narrative "
    "continuity. Continue the passage rather than explaining or summarizing it.\n\nPassage:\n"
)
PG_SUFFIX = "\n\nContinuation:"
DEFAULT_SEED = 42


def fail(message: str) -> None:
    raise ValueError(message)


def digest(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                fail(f"JSONL parse error {path}:{line_no}: {e}")


def parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    return []


def load_parquet(path: Path) -> list[dict]:
    files = parquet_files(path)
    if not files:
        fail(f"No Parquet shards found at {path}")
    # Read the immutable downloaded shards directly.  They are not a
    # Dataset.save_to_disk artifact, so load_from_disk() is intentionally not
    # used here.
    import pyarrow.parquet as pq

    rows = []
    for file in files:
        rows.extend(pq.ParquetFile(file).read().to_pylist())
    return rows


def load_tokenizer(path: str):
    t = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if not getattr(t, "is_fast", False):
        fail("A fast tokenizer is required to resolve RULER token offsets")
    if not getattr(t, "chat_template", None):
        fail("This tokenizer has no chat_template; verify model/tokenizer path")
    return t


def encode(t, text: str) -> list[int]:
    # verbose=False suppresses harmless max_length warnings during PG19 scanning.
    return t.encode(text, add_special_tokens=False, verbose=False)


def render_prompt(t, content: str, assistant_prefix: str = "") -> tuple[str, list[int]]:
    rendered = t.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        fail("chat_template did not return string")
    rendered += assistant_prefix
    ids = encode(t, rendered)
    return rendered, ids


def make_record(
    t,
    *,
    sample_id: str,
    dataset: str,
    task: str,
    user_content: str,
    source_id: str,
    target: int,
    max_new_tokens: int,
    metadata: dict,
    assistant_prefix: str = "",
    reference_answer: Any = None,
    pair_id: str | None = None,
) -> dict:
    prompt, ids = render_prompt(t, user_content, assistant_prefix)
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "task": task,
        "source_id": source_id,
        "pair_id": pair_id,
        "target_prompt_length": target,
        "actual_prompt_length": len(ids),
        "max_new_tokens": max_new_tokens,
        "user_content": user_content,
        "assistant_prefix": assistant_prefix,
        "prompt": prompt,
        "input_ids": ids,
        "input_sha256": digest(json.dumps(ids, separators=(",", ":"))),
        "reference_answer": reference_answer,
        "metadata": metadata,
    }


def load_gov(source: str, cache_dir: str | None = None) -> list[dict]:
    path = Path(source)
    if path.is_file() and path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    if parquet_files(path):
        return load_parquet(path)
    if path.is_dir():
        from datasets import load_from_disk

        obj = load_from_disk(str(path))
        return list(obj["test"] if hasattr(obj, "keys") and "test" in obj else obj)
    if source == "hf-cache":
        from datasets import load_dataset

        return list(
            load_dataset(
                "ccdv/govreport-summarization",
                "document",
                split="test",
                cache_dir=cache_dir,
            )
        )
    fail("GovReport: source must be Parquet, HF save_to_disk directory, JSONL, or 'hf-cache'")


def load_pg19(source: str, fmt: str) -> list[dict]:
    path = Path(source)
    if fmt == "auto":
        if path.is_file() and path.suffix == ".jsonl":
            fmt = "jsonl"
        elif parquet_files(path):
            fmt = "parquet"
        elif path.is_dir() and list(path.glob("*.txt")):
            fmt = "txtdir"
        else:
            fmt = "hf"
    if fmt == "jsonl":
        if not path.is_file():
            fail(f"PG19 JSONL missing: {path}")
        return list(iter_jsonl(path))
    if fmt == "parquet":
        return load_parquet(path)
    if fmt == "txtdir":
        files = sorted(path.glob("*.txt"))
        if not files:
            fail(f"No *.txt files in {path}")
        return [
            {
                "book_id": f.stem,
                "book_title": f.stem,
                "book_text": f.read_text(encoding="utf-8", errors="replace"),
            }
            for f in files
        ]
    if fmt == "hf":
        from datasets import load_from_disk

        obj = load_from_disk(str(path))
        return list(obj["test"] if hasattr(obj, "keys") and "test" in obj else obj)
    if fmt == "tfds":
        import tensorflow_datasets as tfds

        ds = tfds.load(
            "pg19",
            split="test",
            data_dir=str(path),
            download=False,
            shuffle_files=False,
        )
        return list(tfds.as_numpy(ds))
    fail(f"Unsupported pg19 format: {fmt}")


def as_text(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v) if v is not None else ""


def pg_fields(row: dict, idx: int) -> tuple[str, str, str]:
    book_id = as_text(row.get("book_id", row.get("id", row.get("url", idx))))
    title = as_text(row.get("book_title", row.get("short_book_title", row.get("title", book_id))))
    text = as_text(row.get("book_text", row.get("text", "")))
    return book_id, title, text


def gov_content(report: str) -> str:
    return GOV_PREFIX + report + GOV_SUFFIX


def count_gov(t, rows: list[dict]) -> list[dict]:
    result = []
    for i, row in enumerate(rows):
        text = as_text(row.get("report", ""))
        summary = as_text(row.get("summary", ""))
        if not text.strip() or not summary.strip():
            continue
        _, ids = render_prompt(t, gov_content(text))
        result.append(
            {
                "source_index": i,
                "source_id": as_text(row.get("id", i)),
                "prompt_tokens": len(ids),
                "summary_tokens": len(encode(t, summary)),
                "report_chars": len(text),
                "content_sha256": digest(text),
            }
        )
    return result


def length_summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    v = sorted(values)

    def quantile(q: float) -> int:
        return v[round((len(v) - 1) * q)]

    return {
        "count": len(v),
        "min": v[0],
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "max": v[-1],
        "bins": {
            f"{a}-{b}": sum(a <= x < b for x in v)
            for a, b in [
                (7000, 9000),
                (12000, 16000),
                (14000, 18000),
                (20000, 24000),
                (30000, 40000),
            ]
        },
    }


def scan_gov(args) -> None:
    t = load_tokenizer(args.tokenizer)
    items = count_gov(t, load_gov(args.govreport_source, args.hf_cache_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        fail("No valid GovReport records")
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        wr.writeheader()
        wr.writerows(items)
    print(
        json.dumps(
            length_summary([x["prompt_tokens"] for x in items]),
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved {out}")


def target_label(target: int) -> str:
    if target > 0 and target % 1024 == 0:
        return f"{target // 1024}k"
    return str(target)


def select_gov_targets(
    t,
    rows: list[dict],
    targets: list[int],
    n_per_target: int,
    tolerance: int,
    seed: int,
    max_new_tokens: int,
) -> tuple[list[dict], dict]:
    stats = count_gov(t, rows)
    result = []
    report = {
        "total_candidates": len(stats),
        "length_summary": length_summary([x["prompt_tokens"] for x in stats]),
        "targets": {},
    }
    used_content = set()
    for target in sorted(set(targets)):
        eligible = [
            x
            for x in stats
            if abs(x["prompt_tokens"] - target) <= tolerance and x["content_sha256"] not in used_content
        ]
        # Avoid duplicated reports both within and across target-length groups.
        unique = []
        seen = set()
        for x in eligible:
            if x["content_sha256"] not in seen:
                seen.add(x["content_sha256"])
                unique.append(x)
        if len(unique) < n_per_target:
            fail(
                f"GovReport: only {len(unique)} unused samples within "
                f"{target}±{tolerance} tokens, need {n_per_target}; "
                "widen --gov-tolerance or use another target"
            )
        rng = random.Random(f"{seed}:govreport:{target}")
        rng.shuffle(unique)
        chosen = unique[:n_per_target]
        chosen.sort(key=lambda x: (x["prompt_tokens"], x["source_index"]))
        group = target_label(target)
        for i, x in enumerate(chosen, 1):
            row = rows[x["source_index"]]
            result.append(
                make_record(
                    t,
                    sample_id=f"govreport_{group}_{i:03d}",
                    dataset="govreport",
                    task="summarization",
                    user_content=gov_content(as_text(row["report"])),
                    source_id=x["source_id"],
                    target=target,
                    max_new_tokens=max_new_tokens,
                    reference_answer=as_text(row["summary"]),
                    metadata={
                        "source_index": x["source_index"],
                        "report_sha256": x["content_sha256"],
                        "selection_window": [target - tolerance, target + tolerance],
                        "source_split": "test",
                    },
                )
            )
        used_content.update(x["content_sha256"] for x in chosen)
        report["targets"][str(target)] = {
            "eligible_unused": len(unique),
            "selected": n_per_target,
        }
    return result, report


def pg_token_prefix(t, book_text: str, char_limit: int) -> list[int]:
    # Never tokenize an entire million-token book just to construct a 32K example.
    # char_limit is a MAXIMUM, not a truncation of any selected prompt.
    return encode(t, book_text[:char_limit])


def choose_pg_context(t, token_prefix: list[int], anchor: int, target: int, tolerance: int) -> tuple[str, int]:
    # Find a context slice whose *rendered* Llama chat prompt is close to target.
    # Slice end is fixed at anchor; 8K and 32K share the same continuation point.
    lo = 1
    hi = anchor
    best = None
    for _ in range(20):
        if lo > hi:
            break
        n = (lo + hi) // 2
        ctx = t.decode(
            token_prefix[anchor - n : anchor],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        _, ids = render_prompt(t, PG_PREFIX + ctx + PG_SUFFIX)
        delta = abs(len(ids) - target)
        if best is None or delta < best[0]:
            best = (delta, ctx, len(ids))
        if len(ids) < target:
            lo = n + 1
        elif len(ids) > target:
            hi = n - 1
        else:
            break
    if best is None or best[0] > tolerance:
        fail(f"Could not make paired PG19 prompt near {target}±{tolerance}")
    return best[1], best[2]


def select_pg19(
    t,
    books: list[dict],
    n_books: int,
    seed: int,
    targets: tuple[int, int],
    tolerance: int,
    max_new_tokens: int,
    prefix_chars: int,
    future_tokens: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    shuffled = list(enumerate(books))
    rng.shuffle(shuffled)
    chosen = []
    failure = Counter()
    visited = set()
    for original_idx, row in shuffled:
        book_id, title, body = pg_fields(row, original_idx)
        if book_id in visited:
            failure["duplicate_book_id"] += 1
            continue
        visited.add(book_id)
        if not body.strip():
            failure["empty_book"] += 1
            continue
        tokens = pg_token_prefix(t, body, prefix_chars)
        # Need 32K context + reference continuation. Never exceed available tokens.
        if len(tokens) < max(targets) + future_tokens + 256:
            failure["too_short_in_scanned_prefix"] += 1
            continue
        anchor_min = max(targets) + 256
        anchor_high = min(len(tokens) - future_tokens, anchor_min + 8192)
        if anchor_high < anchor_min:
            failure["insufficient_reference_tail"] += 1
            continue
        # Stable per-book random anchor, shared by both length groups.
        anchor = rng.randint(anchor_min, anchor_high)
        try:
            contexts = [choose_pg_context(t, tokens, anchor, target, tolerance) for target in targets]
        except ValueError:
            failure["prompt_budget_not_met"] += 1
            continue
        reference = t.decode(
            tokens[anchor : anchor + future_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not reference.strip():
            failure["empty_reference"] += 1
            continue
        pair_id = f"pg19_{len(chosen)+1:03d}"
        tail = t.decode(
            tokens[anchor - min(anchor, 1024) : anchor],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pair = []
        for group, (target, (context, observed_len)) in zip(("8k", "32k"), zip(targets, contexts)):
            rec = make_record(
                t,
                sample_id=f"{pair_id}_{group}",
                dataset="pg19",
                task="continuation",
                user_content=PG_PREFIX + context + PG_SUFFIX,
                source_id=book_id,
                target=target,
                max_new_tokens=max_new_tokens,
                pair_id=pair_id,
                reference_answer=reference,
                metadata={
                    "source_split": "test",
                    "book_title": title,
                    "source_index": original_idx,
                    "anchor_token_in_scanned_prefix": anchor,
                    "scanned_book_prefix_chars": min(len(body), prefix_chars),
                    "context_tail_sha256": digest(tail),
                    "reference_sha256": digest(reference),
                    "book_text_sha256": digest(body),
                    "context_chars": len(context),
                },
            )
            if abs(rec["actual_prompt_length"] - target) > tolerance:
                fail("Internal PG19 length error")
            pair.append(rec)
        chosen.extend(pair)
        if len(chosen) == n_books * 2:
            break
    if len(chosen) != n_books * 2:
        fail(
            f"Only {len(chosen)//2} eligible PG19 books, need {n_books}. "
            f"Rejection reasons: {dict(failure)}; try increasing --pg19-prefix-chars"
        )
    return chosen, {
        "book_candidates": len(books),
        "selected_books": n_books,
        "rejected_counts": dict(failure),
    }


def resolve_ruler(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = list(p.rglob("validation.jsonl")) + list(p.rglob("test.jsonl"))
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            return candidates[0]
        fail(
            f"RULER path {p} contains {len(candidates)} JSONL files. "
            "Pass the exact .../niah_single_2/validation.jsonl path"
        )
    fail(f"RULER path not found: {p}")


def ruler_value_position(t, source_input: str, answer: str) -> tuple[int, float]:
    # For a single-needle NIAH task the answer appears once in the inserted needle.
    positions = [m.start() for m in re.finditer(re.escape(answer), source_input)]
    if len(positions) != 1:
        fail(f"RULER answer '{answer}' occurs {len(positions)} times in input (expected 1)")
    approx_pos = len(encode(t, source_input[: positions[0]]))
    return approx_pos, approx_pos / max(1, len(encode(t, source_input)))


def select_ruler(
    t,
    path_str: str,
    target: int,
    group: str,
    n: int,
    tolerance: int,
    max_new_tokens: int,
    seed: int,
    task: str,
) -> tuple[list[dict], dict]:
    path = resolve_ruler(path_str)
    if path.parent.name != task:
        fail(f"RULER source task mismatch: expected {task}, found {path.parent.name} " f"at {path}")
    raw = list(iter_jsonl(path))
    candidates = []
    rejects = Counter()
    for idx, item in enumerate(raw):
        content = item.get("input")
        answers = item.get("outputs")
        if not isinstance(content, str) or not isinstance(answers, list) or len(answers) != 1:
            rejects["schema_not_single_needle"] += 1
            continue
        answer = as_text(answers[0])
        if not answer.strip():
            rejects["empty_answer"] += 1
            continue
        try:
            pos, depth = ruler_value_position(t, content, answer)
        except ValueError:
            rejects["answer_not_unique"] += 1
            continue
        answer_prefix = as_text(item.get("answer_prefix", ""))
        _, ids = render_prompt(t, content, answer_prefix)
        if abs(len(ids) - target) > tolerance:
            rejects["prompt_length_outside_window"] += 1
            continue
        candidates.append((idx, item, len(ids), pos, depth))
    if len(candidates) < n:
        fail(
            f"RULER {group}: only {len(candidates)} eligible of {len(raw)}, "
            f"need {n}; rejects={dict(rejects)}; try --ruler-tolerance"
        )
    # Existing 20 generated samples per group are retained when n=20. For larger
    # candidate pools, deterministic shuffling avoids outcome-based selection.
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    selected.sort(key=lambda z: z[4])
    result = []
    for i, (idx, item, length, pos, depth) in enumerate(selected, 1):
        answer = as_text(item["outputs"][0])
        content = item["input"]
        result.append(
            make_record(
                t,
                sample_id=f"ruler_{group}_{i:03d}",
                dataset="ruler",
                task=task,
                user_content=content,
                source_id=f"{path.name}:{idx}",
                target=target,
                max_new_tokens=max_new_tokens,
                assistant_prefix=as_text(item.get("answer_prefix", "")),
                reference_answer=item["outputs"],
                metadata={
                    "source_file": str(path.resolve()),
                    "source_row": idx,
                    "source_split": path.stem,
                    "ruler_task": task,
                    "ruler_original_index": item.get("index"),
                    "source_reported_length": item.get("length"),
                    "source_token_position_answer": item.get("token_position_answer"),
                    "needle_token_position_approx": pos,
                    "needle_relative_depth": round(depth, 6),
                    "needle_occurrences": 1,
                },
            )
        )
    # Do not claim controlled 10/30/... needle depth when the existing pool is random.
    buckets = {
        f"{left}-{left+20}%": sum(left / 100 <= x[4] < (left + 20) / 100 for x in selected)
        for left in (0, 20, 40, 60, 80)
    }
    return result, {
        "source_file": str(path),
        "task": task,
        "raw": len(raw),
        "eligible": len(candidates),
        "selected": len(selected),
        "rejections": dict(rejects),
        "depth_bins": buckets,
        "depth_min": min(x[4] for x in selected),
        "depth_max": max(x[4] for x in selected),
    }


def validate_records(
    records: list[dict],
    t,
    counts: dict[str, int],
    expected_dataset_targets: dict[str, int] | None,
    tolerance_gov: int,
    tolerance_pg: int,
    tolerance_ruler: int,
    max_context: int,
    detailed: bool = True,
) -> dict:
    errors = []
    warnings = []
    seen_ids = set()
    seen_input = set()
    types = Counter()
    dataset_targets = Counter()
    pg_pairs = {}
    sources = {}
    for rec in records:
        sid = rec.get("sample_id", "<missing>")
        if sid in seen_ids:
            errors.append(f"duplicate sample_id {sid}")
        seen_ids.add(sid)
        dataset = rec.get("dataset")
        types[dataset] += 1
        dataset_targets[f"{dataset}:{rec.get('target_prompt_length')}"] += 1
        token_ids = rec.get("input_ids")
        if not isinstance(token_ids, list) or any(not isinstance(x, int) for x in token_ids):
            errors.append(f"{sid}: bad input_ids")
            continue
        prompt = rec.get("prompt", "")
        if not isinstance(prompt, str):
            errors.append(f"{sid}: missing prompt")
            continue
        if len(token_ids) != rec.get("actual_prompt_length"):
            errors.append(f"{sid}: actual_prompt_length mismatch")
        if len(token_ids) + rec.get("max_new_tokens", 0) > max_context:
            errors.append(f"{sid}: prompt + decode budget exceeds context {max_context}")
        if digest(json.dumps(token_ids, separators=(",", ":"))) != rec.get("input_sha256"):
            errors.append(f"{sid}: token hash mismatch")
        if detailed:
            if encode(t, prompt) != token_ids:
                errors.append(f"{sid}: tokenizer(prompt) does not match input_ids")
            expected, _ = render_prompt(t, rec.get("user_content", ""), rec.get("assistant_prefix", ""))
            if expected != prompt:
                errors.append(f"{sid}: chat template mismatch")
        if rec["input_sha256"] in seen_input:
            errors.append(f"{sid}: duplicate input_ids")
        seen_input.add(rec["input_sha256"])
        target = rec.get("target_prompt_length", 0)
        tol = {
            "govreport": tolerance_gov,
            "pg19": tolerance_pg,
            "ruler": tolerance_ruler,
        }.get(dataset, 0)
        if abs(len(token_ids) - target) > tol:
            errors.append(f"{sid}: token length {len(token_ids)} outside {target}±{tol}")
        if dataset == "ruler":
            answers = rec.get("reference_answer")
            if not isinstance(answers, list) or len(answers) != 1:
                errors.append(f"{sid}: expected one RULER answer")
            elif rec.get("user_content", "").count(as_text(answers[0])) != 1:
                errors.append(f"{sid}: RULER needle answer is not unique in user content")
        if dataset == "pg19":
            pg_pairs.setdefault(rec.get("pair_id"), []).append(rec)
        if dataset == "govreport" and not rec.get("reference_answer"):
            errors.append(f"{sid}: missing GovReport reference summary")
        if dataset == "govreport":
            source = rec["source_id"]
            if source in sources:
                errors.append(f"{sid}: repeated GovReport source_id {source}")
            sources[source] = sid
    for name, expected in counts.items():
        if types[name] != expected:
            errors.append(f"{name}: got {types[name]}, expected {expected}")
    if expected_dataset_targets is not None:
        for name, expected in expected_dataset_targets.items():
            actual = dataset_targets[name]
            if actual != expected:
                errors.append(f"{name}: got {actual}, expected {expected}")
    for pair_id, pair in pg_pairs.items():
        if not pair_id or len(pair) != 2:
            errors.append(f"PG19 pair {pair_id} has {len(pair)} records, expected 2")
            continue
        a, b = pair
        if a["source_id"] != b["source_id"]:
            errors.append(f"PG19 {pair_id}: different book sources")
        for fld in (
            "anchor_token_in_scanned_prefix",
            "context_tail_sha256",
            "reference_sha256",
        ):
            if a["metadata"].get(fld) != b["metadata"].get(fld):
                errors.append(f"PG19 {pair_id}: {fld} differs")
        if a.get("reference_answer") != b.get("reference_answer"):
            errors.append(f"PG19 {pair_id}: reference continuation differs")
        if {a["target_prompt_length"], b["target_prompt_length"]} != {8192, 32768}:
            warnings.append(f"PG19 {pair_id}: nonstandard length targets")
        short = min(pair, key=lambda p: len(p["user_content"]))
        long = max(pair, key=lambda p: len(p["user_content"]))
        # Continuation context is the same suffix (except tiny tokenizer boundary artifacts).
        short_ctx = short["user_content"][len(PG_PREFIX) : -len(PG_SUFFIX)]
        long_ctx = long["user_content"][len(PG_PREFIX) : -len(PG_SUFFIX)]
        if short_ctx[-min(1000, len(short_ctx)) :] != long_ctx[-min(1000, len(short_ctx)) :]:
            errors.append(f"PG19 {pair_id}: last 1000 characters not identical")
    return {
        "passed": not errors,
        "counts": dict(types),
        "dataset_target_counts": dict(dataset_targets),
        "total": len(records),
        "errors": errors,
        "warnings": warnings,
        "lengths": {
            name: length_summary([r["actual_prompt_length"] for r in records if r["dataset"] == name])
            for name in counts
        },
        "pg19_pairs": len(pg_pairs),
    }


def write_selection_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "sample_id",
        "dataset",
        "task",
        "source_id",
        "pair_id",
        "target_prompt_length",
        "actual_prompt_length",
        "max_new_tokens",
        "needle_relative_depth",
        "anchor_token",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for rec in records:
            meta = rec.get("metadata", {})
            writer.writerow(
                {
                    **{col: rec.get(col, "") for col in cols},
                    "needle_relative_depth": meta.get("needle_relative_depth", ""),
                    "anchor_token": meta.get("anchor_token_in_scanned_prefix", ""),
                }
            )


def make_manifest(path: Path, records: list[dict], t, args, audit: dict, sources: dict) -> dict:
    return {
        "schema_version": "head-analysis-dataset-v2",
        "selection_seed": args.seed,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_class": type(t).__name__,
        "vocab_size": len(t),
        "model_context_limit": args.max_context,
        "gov_targets": sorted(args.gov_targets),
        "gov_count_per_length": args.gov_count_per_length,
        "gov_tolerance": args.gov_tolerance,
        "pg_targets": [8192, 32768],
        "pg_tolerance": args.pg_tolerance,
        "ruler_tolerance": args.ruler_tolerance,
        "ruler_task": args.ruler_task,
        "sources": sources,
        "counts": audit["counts"],
        "dataset_target_counts": audit["dataset_target_counts"],
        "data_file": "prompts.jsonl",
        "data_sha256": digest(path.read_bytes()),
        "template_sha256": {
            "govreport": digest(GOV_PREFIX + GOV_SUFFIX),
            "pg19": digest(PG_PREFIX + PG_SUFFIX),
        },
        "audit_passed": audit["passed"],
    }


def build(args) -> None:
    t = load_tokenizer(args.tokenizer)
    print("[1/4] Reading GovReport...")
    gov, gov_report = select_gov_targets(
        t,
        load_gov(args.govreport_source, args.hf_cache_dir),
        args.gov_targets,
        args.gov_count_per_length,
        args.gov_tolerance,
        args.seed,
        args.gov_decode,
    )
    print("[2/4] Reading PG19 and building paired contexts...")
    pg, pg_report = select_pg19(
        t,
        load_pg19(args.pg19_source, args.pg19_format),
        args.pg_books,
        args.seed,
        (8192, 32768),
        args.pg_tolerance,
        args.pg_decode,
        args.pg19_prefix_chars,
        args.pg_reference_tokens,
    )
    print("[3/4] Validating RULER 8K and 32K candidates...")
    r8, r8_report = select_ruler(
        t,
        args.ruler_8k,
        8192,
        "8k",
        args.ruler_count_per_length,
        args.ruler_tolerance,
        args.ruler_decode,
        args.seed,
        args.ruler_task,
    )
    r32, r32_report = select_ruler(
        t,
        args.ruler_32k,
        32768,
        "32k",
        args.ruler_count_per_length,
        args.ruler_tolerance,
        args.ruler_decode,
        args.seed,
        args.ruler_task,
    )
    records = gov + pg + r8 + r32
    counts = {
        "govreport": len(set(args.gov_targets)) * args.gov_count_per_length,
        "pg19": args.pg_books * 2,
        "ruler": args.ruler_count_per_length * 2,
    }
    expected_dataset_targets = {
        **{f"govreport:{target}": args.gov_count_per_length for target in set(args.gov_targets)},
        "pg19:8192": args.pg_books,
        "pg19:32768": args.pg_books,
        "ruler:8192": args.ruler_count_per_length,
        "ruler:32768": args.ruler_count_per_length,
    }
    print("[4/4] Running full validation...")
    audit = validate_records(
        records,
        t,
        counts,
        expected_dataset_targets,
        args.gov_tolerance,
        args.pg_tolerance,
        args.ruler_tolerance,
        args.max_context,
    )
    audit["selection"] = {
        "govreport": gov_report,
        "pg19": pg_report,
        "ruler_8k": r8_report,
        "ruler_32k": r32_report,
    }
    out = Path(args.output)
    if (out / "prompts.jsonl").exists() or (out / "manifest.json").exists():
        fail(f"Final dataset already exists in {out}; use a new versioned output directory")
    out.mkdir(parents=True, exist_ok=True)
    json_dump(audit, out / "selection_report.json")
    if not audit["passed"]:
        print(json.dumps(audit["errors"], indent=2, ensure_ascii=False), file=sys.stderr)
        fail("Dataset validation failed; no final dataset exported. See selection_report.json")
    write_selection_csv(records, out / "selected_samples.csv")
    for name, subset in (("govreport", gov), ("pg19", pg), ("ruler", r8 + r32)):
        write_jsonl(subset, out / f"{name}.jsonl")
    all_file = out / "prompts.jsonl"
    write_jsonl(records, all_file)
    sources = {
        "govreport_source": args.govreport_source,
        "pg19_source": args.pg19_source,
        "pg19_format": args.pg19_format,
        "ruler_task": args.ruler_task,
        "ruler_8k": str(resolve_ruler(args.ruler_8k)),
        "ruler_32k": str(resolve_ruler(args.ruler_32k)),
    }
    manifest = make_manifest(all_file, records, t, args, audit, sources)
    json_dump(manifest, out / "manifest.json")
    print(
        json.dumps(
            {
                "passed": True,
                "counts": audit["counts"],
                "file": str(all_file),
                "sha256": manifest["data_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def verify(args) -> None:
    inp = Path(args.input)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = list(iter_jsonl(inp))
    counts = manifest["counts"]
    if digest(inp.read_bytes()) != manifest["data_sha256"]:
        fail("prompts.jsonl checksum differs from manifest; file was modified")
    t = load_tokenizer(args.tokenizer)
    results = validate_records(
        records,
        t,
        counts,
        manifest.get("dataset_target_counts"),
        manifest["gov_tolerance"],
        manifest["pg_tolerance"],
        manifest["ruler_tolerance"],
        manifest["model_context_limit"],
    )
    json_dump(results, inp.parent / "validation_report.json")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if not results["passed"]:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan-gov", help="Write GovReport candidate lengths CSV")
    scan.add_argument("--tokenizer", required=True)
    scan.add_argument("--govreport-source", required=True)
    scan.add_argument("--hf-cache-dir")
    scan.add_argument("--output", required=True)
    scan.set_defaults(func=scan_gov)

    b = sub.add_parser("build", help="Create and validate fixed prompt dataset")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--govreport-source", required=True)
    b.add_argument("--hf-cache-dir")
    b.add_argument("--pg19-source", required=True)
    b.add_argument(
        "--pg19-format",
        choices=["auto", "hf", "tfds", "jsonl", "parquet", "txtdir"],
        default="auto",
    )
    b.add_argument("--ruler-8k", required=True)
    b.add_argument("--ruler-32k", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--gov-targets", type=int, nargs="+", default=[8192, 16384])
    b.add_argument("--gov-count-per-length", type=int, default=20)
    b.add_argument("--gov-tolerance", type=int, default=1024)
    b.add_argument("--gov-decode", type=int, default=1024)
    b.add_argument("--pg-books", type=int, default=10)
    b.add_argument("--pg-tolerance", type=int, default=64)
    b.add_argument("--pg-decode", type=int, default=1024)
    b.add_argument("--pg-reference-tokens", type=int, default=1024)
    b.add_argument("--pg19-prefix-chars", type=int, default=320000)
    b.add_argument("--ruler-count-per-length", type=int, default=20)
    b.add_argument("--ruler-tolerance", type=int, default=512)
    b.add_argument("--ruler-decode", type=int, default=128)
    b.add_argument("--ruler-task", default="niah_single_2")
    b.add_argument("--max-context", type=int, default=131072)
    b.set_defaults(func=build)

    v = sub.add_parser("verify", help="Recheck exported dataset from saved input_ids")
    v.add_argument("--tokenizer", required=True)
    v.add_argument("--input", required=True)
    v.add_argument("--manifest", required=True)
    v.set_defaults(func=verify)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
