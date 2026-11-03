from __future__ import annotations

from pathlib import Path

import pytest
from minisgl.moe.offload import (
    ExpertEvictionProfile,
    ExpertOffloadController,
    ExpertProfileEntry,
    ExpertResidencyState,
    KVPressureSnapshot,
)


def _profile(*frequencies: float) -> ExpertEvictionProfile:
    return ExpertEvictionProfile(
        [
            ExpertProfileEntry(layer_id=0, expert_id=i, activation_frequency=f)
            for i, f in enumerate(frequencies)
        ]
    )


def test_real_profile_matches_documented_shape_and_order() -> None:
    path = Path(__file__).parents[2] / "coldest_experts.txt"
    profile = ExpertEvictionProfile.from_file(path)

    assert len(profile.entries) == 960
    assert len(profile.allowed_keys) == 960
    assert {entry.layer_id for entry in profile.entries} == set(range(48))
    assert all(
        sum(entry.layer_id == layer_id for entry in profile.entries) == 20 for layer_id in range(48)
    )
    assert profile.entries[0].key == (45, 97)
    assert profile.entries[-1].key == (0, 117)


def test_profile_rejects_unsorted_and_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="sorted"):
        ExpertEvictionProfile(
            [
                ExpertProfileEntry(0, 0, 0.2),
                ExpertProfileEntry(0, 1, 0.1),
            ]
        )

    with pytest.raises(ValueError, match="Duplicate"):
        ExpertEvictionProfile(
            [
                ExpertProfileEntry(0, 0, 0.1),
                ExpertProfileEntry(0, 0, 0.2),
            ]
        )


def test_eviction_plan_follows_profile_until_capacity_is_sufficient() -> None:
    profile = _profile(0.01, 0.02, 0.03)
    controller = ExpertOffloadController(profile)
    pressure = KVPressureSnapshot(
        persistent_available_blocks=2,
        temporary_available_blocks=1,
        upcoming_demand_blocks=10,
    )
    capacities = {(0, 0): 3, (0, 1): 5, (0, 2): 100}

    plan = controller.plan_evictions(pressure, capacities)

    assert [entry.key for entry in plan.experts] == [(0, 0), (0, 1)]
    assert plan.recovered_blocks == 8
    assert plan.remaining_shortfall_blocks == 0
    assert plan.projected_cumulative_frequency == pytest.approx(0.03)
    assert controller.cumulative_offloaded_frequency == 0


def test_all_candidates_can_be_evicted_without_frequency_cap() -> None:
    profile = _profile(0.01, 0.02, 0.03)
    controller = ExpertOffloadController(profile)
    pressure = KVPressureSnapshot(0, 0, 10)

    plan = controller.plan_evictions(
        pressure,
        {(0, 0): 3, (0, 1): 3, (0, 2): 4},
    )

    assert [entry.key for entry in plan.experts] == [(0, 0), (0, 1), (0, 2)]
    assert plan.remaining_shortfall_blocks == 0
    assert plan.projected_cumulative_frequency == pytest.approx(0.06)


def test_restoration_does_not_require_draining_or_reverse_order() -> None:
    profile = _profile(0.01, 0.02, 0.03)
    controller = ExpertOffloadController(profile)

    for key in ((0, 0), (0, 2)):
        controller.begin_eviction(key)
        controller.complete_eviction(key)

    assert controller.cumulative_offloaded_frequency == pytest.approx(0.04)
    controller.begin_restoring((0, 0))
    controller.complete_restoration((0, 0))

    assert controller.residency.state((0, 0)) is ExpertResidencyState.GPU_RESIDENT
    assert controller.residency.state((0, 2)) is ExpertResidencyState.CPU_RESIDENT
    assert controller.cumulative_offloaded_frequency == pytest.approx(0.03)


def test_non_candidate_and_invalid_state_transitions_are_rejected() -> None:
    controller = ExpertOffloadController(_profile(0.01))

    with pytest.raises(KeyError, match="not allowed"):
        controller.begin_eviction((9, 9))

    with pytest.raises(RuntimeError, match="Invalid residency transition"):
        controller.complete_eviction((0, 0))


def test_stop_hook_applies_to_plan_and_execution() -> None:
    class StoppedController(ExpertOffloadController):
        def check_stop_moe_eviction(self) -> bool:
            return True

    controller = StoppedController(_profile(0.01))
    assert not controller.plan_evictions(KVPressureSnapshot(0, 0, 1), {(0, 0): 1}).experts
    with pytest.raises(RuntimeError, match="stopped by policy"):
        controller.begin_eviction((0, 0))
