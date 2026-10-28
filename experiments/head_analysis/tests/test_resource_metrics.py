import pytest
import torch

from experiments.head_analysis.runners.resource_metrics import (
    GenerationPhaseTimer,
    ResourceMeasurementError,
    validate_standard_greedy_timing,
)


def test_cpu_phase_timer_records_prefill_then_decode_and_removes_hooks():
    model = torch.nn.Linear(3, 3).eval()
    with torch.inference_mode(), GenerationPhaseTimer(model, torch.device("cpu")) as timer:
        model(torch.ones(1, 3))
        model(torch.ones(1, 3))
        model(torch.ones(1, 3))

    summary = timer.summary()
    assert summary["method"] == "host_perf_counter"
    assert summary["completed_forward_count"] == 3
    assert summary["prefill"]["count"] == 1
    assert summary["decode"]["count"] == 2
    assert summary["decode"]["total_seconds"] >= 0
    assert summary["decode"]["p50_seconds"] is not None
    assert summary["decode"]["p95_seconds"] is not None
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    validate_standard_greedy_timing(summary, generated_length=3)


def test_phase_timer_rejects_trace_length_that_disagrees_with_forwards():
    model = torch.nn.Linear(2, 2).eval()
    with GenerationPhaseTimer(model, torch.device("cpu")) as timer:
        model(torch.ones(1, 2))
        model(torch.ones(1, 2))

    with pytest.raises(ResourceMeasurementError, match="standard greedy timing"):
        validate_standard_greedy_timing(timer.summary(), generated_length=4)
