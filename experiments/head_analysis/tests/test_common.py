import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.head_analysis.common.io import atomic_file, iter_jsonl, read_json, write_json, write_npz
from experiments.head_analysis.common.statistics import bootstrap_sample_mean_ci, finite_mean_count


def test_common_imports_without_loading_model_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from experiments.head_analysis.common import io, statistics; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_jsonl_reader_preserves_object_option_and_line_errors(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text":"中文"}\n\n[1,2]\n', encoding="utf-8")
    assert list(iter_jsonl(path)) == [{"text": "中文"}, [1, 2]]
    with pytest.raises(ValueError, match="data.jsonl:3 is not an object"):
        list(iter_jsonl(path, require_objects=True))
    path.write_text('{}\n{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL parse error .*data.jsonl:2"):
        list(iter_jsonl(path))


def test_shared_writes_roundtrip_and_preserve_existing_files(tmp_path):
    text, numeric = tmp_path / "nested/data.json", tmp_path / "nested/data.npz"
    write_json(text, {"样本": 2})
    assert read_json(text) == {"样本": 2}
    values = np.array([1, 2, 3], dtype=np.uint32)
    write_npz(numeric, {"ids": values})
    with np.load(numeric) as archive:
        np.testing.assert_array_equal(archive["ids"], values)
    for path, write, payload in [(text, write_json, {}), (numeric, write_npz, {"ids": values + 1})]:
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            write(path, payload)
        assert path.read_bytes() == before
    assert not list(text.parent.glob(".*.tmp"))


def test_failed_write_keeps_previous_output_and_cleans_temporary_file(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"previous": True})
    with pytest.raises(RuntimeError):
        with atomic_file(path, overwrite=True) as handle:
            handle.write("partial")
            raise RuntimeError("serialization failed")
    assert read_json(path) == {"previous": True}
    with pytest.raises(ValueError):
        write_json(path, {"invalid": np.nan}, overwrite=True)
    assert json.loads(path.read_text()) == {"previous": True}
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_finite_means_keep_requested_precision_and_missing_counts(dtype):
    values = np.array([[1, np.nan, 3], [np.inf, np.nan, -2]], dtype=np.float32)
    means, counts = finite_mean_count(values, 0, dtype=dtype, count_dtype=np.uint32)
    np.testing.assert_array_equal(counts, [1, 0, 2])
    np.testing.assert_allclose(means, [1, np.nan, 0.5], equal_nan=True)
    assert means.dtype == dtype and counts.dtype == np.uint32


def test_bootstrap_sample_mean_keeps_b1_draw_order_and_missing_values():
    values = np.array(
        [[1, np.nan, 2, np.nan], [2, 4, np.nan, np.nan], [6, 5, 8, np.nan], [np.nan, 7, 5, np.nan]],
        dtype=np.float32,
    )
    rng = np.random.default_rng(31)
    ci = bootstrap_sample_mean_ci(values, 37, rng)
    np.testing.assert_array_equal(ci, np.array([[1.3, 6], [4, 6.55], [2, 8], [np.nan, np.nan]], dtype=np.float32))
    assert ci.dtype == np.float32
    assert rng.integers(0, 1_000_000) == 56575


def test_extraction_preserves_both_existing_missing_head_centering_policies():
    from experiments.head_analysis.analysis import metric_relationships as b1
    from experiments.head_analysis.analysis.b2_relationships import correlate_profiles

    x = np.array([[0, 3, 9, 14], [14, 8, 10, 17]], dtype=float)
    y = np.array([[18, np.nan, 11, 9], [10, 11, 18, 19]], dtype=float)
    old_policy = b1._correlation(b1._layer_center(x).ravel(), b1._layer_center(y).ravel())
    pairwise_policy = correlate_profiles(x, y)
    assert old_policy[1] == pytest.approx(-0.35714285714285715)
    assert pairwise_policy["values"][1, 0] == pytest.approx(-0.4642857142857143)
