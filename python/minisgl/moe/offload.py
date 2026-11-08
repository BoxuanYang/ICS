from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeAlias

ExpertKey: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ExpertProfileEntry:
    layer_id: int
    expert_id: int
    activation_frequency: float

    @property
    def key(self) -> ExpertKey:
        return (self.layer_id, self.expert_id)


class ExpertResidencyState(Enum):
    GPU_RESIDENT = auto()
    EVICTING = auto()
    CPU_RESIDENT = auto()
    RESTORING = auto()


_ALLOWED_TRANSITIONS = {
    ExpertResidencyState.GPU_RESIDENT: {ExpertResidencyState.EVICTING},
    ExpertResidencyState.EVICTING: {ExpertResidencyState.CPU_RESIDENT},
    ExpertResidencyState.CPU_RESIDENT: {ExpertResidencyState.RESTORING},
    ExpertResidencyState.RESTORING: {ExpertResidencyState.GPU_RESIDENT},
}


class ExpertEvictionProfile:
    """Validated, ordered expert-eviction candidates from offline profiling."""

    def __init__(self, entries: Sequence[ExpertProfileEntry]) -> None:
        if not entries:
            raise ValueError("Expert eviction profile must not be empty")

        previous_frequency = -math.inf
        entry_by_key: dict[ExpertKey, ExpertProfileEntry] = {}
        for index, entry in enumerate(entries):
            if isinstance(entry.layer_id, bool) or not isinstance(entry.layer_id, int):
                raise TypeError(f"Entry {index} has a non-integer layer_id")
            if isinstance(entry.expert_id, bool) or not isinstance(entry.expert_id, int):
                raise TypeError(f"Entry {index} has a non-integer expert_id")
            if entry.layer_id < 0 or entry.expert_id < 0:
                raise ValueError(f"Entry {index} has a negative expert key: {entry.key}")
            if not math.isfinite(entry.activation_frequency):
                raise ValueError(f"Entry {index} has a non-finite activation frequency")
            if entry.activation_frequency < 0:
                raise ValueError(f"Entry {index} has a negative activation frequency")
            if entry.activation_frequency < previous_frequency:
                raise ValueError(
                    "Expert eviction profile must be sorted by activation frequency "
                    f"(entry {index - 1} > entry {index})"
                )
            if entry.key in entry_by_key:
                raise ValueError(f"Duplicate expert key in profile: {entry.key}")
            entry_by_key[entry.key] = entry
            previous_frequency = entry.activation_frequency

        self._entries = tuple(entries)
        self._entry_by_key = entry_by_key

    @classmethod
    def from_file(cls, path: str | Path) -> ExpertEvictionProfile:
        profile_path = Path(path)
        try:
            raw_entries = ast.literal_eval(profile_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError) as exc:
            raise ValueError(f"Unable to parse expert profile {profile_path}: {exc}") from exc

        if not isinstance(raw_entries, (list, tuple)):
            raise ValueError("Expert profile must be a list or tuple")

        entries: list[ExpertProfileEntry] = []
        for index, raw_entry in enumerate(raw_entries):
            if isinstance(raw_entry, Mapping):
                try:
                    layer_id = raw_entry["layer_id"]
                    expert_id = raw_entry["expert_id"]
                    frequency = raw_entry["activation_frequency"]
                except KeyError as exc:
                    raise ValueError(
                        f"Profile entry {index} is missing field {exc.args[0]!r}"
                    ) from exc
            elif isinstance(raw_entry, (list, tuple)) and len(raw_entry) == 3:
                layer_id, expert_id, frequency = raw_entry
            else:
                raise ValueError(
                    f"Profile entry {index} must be a mapping or a "
                    "(layer, expert, frequency) triple"
                )
            if isinstance(layer_id, bool) or not isinstance(layer_id, int):
                raise TypeError(f"Profile entry {index} has a non-integer layer_id")
            if isinstance(expert_id, bool) or not isinstance(expert_id, int):
                raise TypeError(f"Profile entry {index} has a non-integer expert_id")
            if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
                raise TypeError(f"Profile entry {index} has a non-numeric activation frequency")
            entries.append(ExpertProfileEntry(layer_id, expert_id, float(frequency)))
        return cls(entries)

    def prefix(self, count: int) -> ExpertEvictionProfile:
        """Return the `count` coldest entries while preserving profile order."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Expert profile prefix count must be an integer")
        if count < 1:
            raise ValueError("Expert profile prefix count must be positive")
        if count >= len(self._entries):
            return self
        return ExpertEvictionProfile(self._entries[:count])

    @property
    def entries(self) -> tuple[ExpertProfileEntry, ...]:
        return self._entries

    @property
    def allowed_keys(self) -> frozenset[ExpertKey]:
        return frozenset(self._entry_by_key)

    def entry(self, key: ExpertKey) -> ExpertProfileEntry:
        try:
            return self._entry_by_key[key]
        except KeyError as exc:
            raise KeyError(f"Expert {key} is not an eviction candidate") from exc


class ExpertResidencyTable:
    """Residency state for profile candidates; non-candidates remain GPU resident."""

    def __init__(self, allowed_keys: Iterable[ExpertKey]) -> None:
        self._allowed_keys = frozenset(allowed_keys)
        self._states: dict[ExpertKey, ExpertResidencyState] = {}

    def state(self, key: ExpertKey) -> ExpertResidencyState:
        return self._states.get(key, ExpertResidencyState.GPU_RESIDENT)

    def transition(self, key: ExpertKey, new_state: ExpertResidencyState) -> None:
        if key not in self._allowed_keys:
            raise KeyError(f"Expert {key} is not allowed to leave GPU residency")
        old_state = self.state(key)
        if new_state not in _ALLOWED_TRANSITIONS[old_state]:
            raise RuntimeError(
                f"Invalid residency transition for expert {key}: "
                f"{old_state.name} -> {new_state.name}"
            )
        if new_state is ExpertResidencyState.GPU_RESIDENT:
            self._states.pop(key, None)
        else:
            self._states[key] = new_state

    def keys_in(self, *states: ExpertResidencyState) -> frozenset[ExpertKey]:
        wanted = frozenset(states)
        return frozenset(key for key, state in self._states.items() if state in wanted)


@dataclass(frozen=True, slots=True)
class KVPressureSnapshot:
    persistent_available_blocks: int
    temporary_available_blocks: int
    upcoming_demand_blocks: int

    def __post_init__(self) -> None:
        values = (
            self.persistent_available_blocks,
            self.temporary_available_blocks,
            self.upcoming_demand_blocks,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("KV pressure values must be integers")
        if any(value < 0 for value in values):
            raise ValueError("KV pressure values must be non-negative")

    @property
    def shortfall_blocks(self) -> int:
        available = self.persistent_available_blocks + self.temporary_available_blocks
        return max(0, self.upcoming_demand_blocks - available)


@dataclass(frozen=True, slots=True)
class EvictionPlan:
    experts: tuple[ExpertProfileEntry, ...]
    recovered_blocks: int
    remaining_shortfall_blocks: int
    projected_cumulative_frequency: float


class ExpertOffloadController:
    """Profile-ordered eviction. Restoration is driven by buffer release events."""

    _OFFLOADED_STATES = frozenset(
        {
            ExpertResidencyState.EVICTING,
            ExpertResidencyState.CPU_RESIDENT,
            ExpertResidencyState.RESTORING,
        }
    )

    def __init__(
        self,
        profile: ExpertEvictionProfile,
    ) -> None:
        self.profile = profile
        self.residency = ExpertResidencyTable(profile.allowed_keys)

    def check_stop_moe_eviction(self) -> bool:
        """Extension point for future SLO policy; no frequency cap in the demo."""
        return False

    @property
    def cumulative_offloaded_frequency(self) -> float:
        return math.fsum(
            entry.activation_frequency
            for entry in self.profile.entries
            if self.residency.state(entry.key) in self._OFFLOADED_STATES
        )

    def plan_evictions(
        self,
        pressure: KVPressureSnapshot,
        capacity_blocks: Mapping[ExpertKey, int],
    ) -> EvictionPlan:
        remaining = pressure.shortfall_blocks
        current_frequency = self.cumulative_offloaded_frequency
        projected_frequency = current_frequency
        selected: list[ExpertProfileEntry] = []
        recovered_blocks = 0

        if remaining == 0:
            return EvictionPlan((), 0, 0, current_frequency)

        for entry in self.profile.entries:
            if self.check_stop_moe_eviction():
                break
            if self.residency.state(entry.key) is not ExpertResidencyState.GPU_RESIDENT:
                continue
            capacity = capacity_blocks.get(entry.key, 0)
            if isinstance(capacity, bool) or not isinstance(capacity, int):
                raise TypeError(f"Capacity for expert {entry.key} must be an integer")
            if capacity < 0:
                raise ValueError(f"Capacity for expert {entry.key} must be non-negative")

            next_frequency = projected_frequency + entry.activation_frequency
            if capacity == 0:
                continue

            selected.append(entry)
            recovered_blocks += capacity
            remaining = max(0, remaining - capacity)
            projected_frequency = next_frequency
            if remaining == 0:
                break

        return EvictionPlan(
            experts=tuple(selected),
            recovered_blocks=recovered_blocks,
            remaining_shortfall_blocks=remaining,
            projected_cumulative_frequency=projected_frequency,
        )

    def begin_eviction(self, key: ExpertKey) -> None:
        if self.check_stop_moe_eviction():
            raise RuntimeError("MoE eviction stopped by policy")
        self.residency.transition(key, ExpertResidencyState.EVICTING)

    def complete_eviction(self, key: ExpertKey) -> None:
        self.residency.transition(key, ExpertResidencyState.CPU_RESIDENT)

    def begin_restoring(self, key: ExpertKey) -> None:
        self.residency.transition(key, ExpertResidencyState.RESTORING)

    def complete_restoration(self, key: ExpertKey) -> None:
        self.residency.transition(key, ExpertResidencyState.GPU_RESIDENT)
