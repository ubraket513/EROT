"""Allocation-safe CPU/GPU worker plans without requiring accelerator hardware."""

import pytest

from erot.runtime.environment import resource_plan


def test_cpu_workers_have_disjoint_affinity_and_thread_budgets():
    plan = resource_plan(2, "cpu", affinity=list(range(8)), environ={})
    assert [p["cpus"] for p in plan] == [list(range(4)), list(range(4, 8))]
    assert all(p["threads"] == 4 and p["gpu_token"] is None for p in plan)


def test_scheduler_budget_and_explicit_thread_limits():
    plan = resource_plan(
        2,
        "cpu",
        affinity=list(range(16)),
        environ={"SLURM_CPUS_PER_TASK": "6"},
        threads_per_worker=2,
    )
    assert [p["cpus"] for p in plan] == [[0, 1, 2], [3, 4, 5]]
    assert [p["threads"] for p in plan] == [2, 2]
    with pytest.raises(ValueError, match="CPU"):
        resource_plan(
            2,
            "cpu",
            affinity=list(range(16)),
            environ={"SLURM_CPUS_PER_TASK": "6"},
            cpu_budget=8,
        )


def test_gpu_tokens_and_auto_worker_count():
    plan = resource_plan(
        None,
        "gpu",
        affinity=list(range(8)),
        environ={"CUDA_VISIBLE_DEVICES": "GPU-abc,MIG-def"},
    )
    assert [p["gpu_token"] for p in plan] == ["GPU-abc", "MIG-def"]
    assert set(plan[0]["cpus"]).isdisjoint(plan[1]["cpus"])


@pytest.mark.parametrize(
    "options",
    [
        {"workers": 9},
        {"workers": 2, "threads_per_worker": 5},
        {"workers": 0},
        {"workers": 1.5},
        {"workers": 2, "cpu_budget": 1},
        {"workers": 2, "device": "gpu", "gpu_tokens": ["GPU-a"]},
        {"workers": 2, "device": "gpu", "gpu_tokens": ["GPU-a", "GPU-a"]},
    ],
)
def test_oversubscribed_or_invalid_plans_rejected(options):
    with pytest.raises(ValueError):
        resource_plan(
            **({"workers": 2, "device": "cpu"} | options),
            affinity=list(range(8)),
            environ={},
        )


def test_gpu_allocation_must_be_explicit_or_visible():
    with pytest.raises(ValueError, match="GPU"):
        resource_plan(1, "gpu", affinity=[0, 1], environ={})
