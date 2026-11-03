from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

from minisgl.kvcache.unified import TemporaryKVRegion, UnifiedKVAllocator

from .offload import ExpertEvictionProfile, ExpertKey


class TensorStorageView(Protocol):
    def numel(self) -> int: ...

    def element_size(self) -> int: ...


class MoELayerStorageView(Protocol):
    layer_id: int

    def expert_tensor_views(
        self, expert_id: int
    ) -> tuple[TensorStorageView, TensorStorageView]: ...


@dataclass(slots=True)
class ExpertStorage:
    layer_id: int
    expert_id: int
    activation_frequency: float
    gate_up_tensor: TensorStorageView
    down_tensor: TensorStorageView
    gate_up_region: TemporaryKVRegion | None = None
    down_region: TemporaryKVRegion | None = None

    @property
    def key(self) -> ExpertKey:
        return (self.layer_id, self.expert_id)

    @property
    def gate_up_nbytes(self) -> int:
        return self.gate_up_tensor.numel() * self.gate_up_tensor.element_size()

    @property
    def down_nbytes(self) -> int:
        return self.down_tensor.numel() * self.down_tensor.element_size()

    @property
    def total_nbytes(self) -> int:
        return self.gate_up_nbytes + self.down_nbytes

    def create_temporary_regions(
        self, block_nbytes: int
    ) -> tuple[TemporaryKVRegion, TemporaryKVRegion]:
        if self.gate_up_region is not None or self.down_region is not None:
            raise RuntimeError(f"Temporary regions already exist for expert {self.key}")

        prefix = f"expert-{self.layer_id}-{self.expert_id}"
        self.gate_up_region = TemporaryKVRegion(
            region_id=f"{prefix}-gate-up",
            owner_expert=self.key,
            storage_nbytes=self.gate_up_nbytes,
            block_nbytes=block_nbytes,
            storage_view=self.gate_up_tensor,
        )
        self.down_region = TemporaryKVRegion(
            region_id=f"{prefix}-down",
            owner_expert=self.key,
            storage_nbytes=self.down_nbytes,
            block_nbytes=block_nbytes,
            storage_view=self.down_tensor,
        )
        return self.gate_up_region, self.down_region


class ExpertStorageCatalog:
    def __init__(self, storages: Iterable[ExpertStorage]) -> None:
        self._storages: dict[ExpertKey, ExpertStorage] = {}
        for storage in storages:
            if storage.key in self._storages:
                raise ValueError(f"Duplicate expert storage: {storage.key}")
            self._storages[storage.key] = storage

    @classmethod
    def from_layers(
        cls,
        layers: Iterable[MoELayerStorageView],
        profile: ExpertEvictionProfile,
    ) -> ExpertStorageCatalog:
        layer_by_id: dict[int, MoELayerStorageView] = {}
        for layer in layers:
            if layer.layer_id in layer_by_id:
                raise ValueError(f"Duplicate MoE layer id: {layer.layer_id}")
            layer_by_id[layer.layer_id] = layer

        storages: list[ExpertStorage] = []
        for entry in profile.entries:
            try:
                layer = layer_by_id[entry.layer_id]
            except KeyError as exc:
                raise ValueError(f"Profile references missing MoE layer {entry.layer_id}") from exc
            try:
                gate_up, down = layer.expert_tensor_views(entry.expert_id)
            except IndexError as exc:
                raise ValueError(f"Profile references missing expert {entry.key}") from exc
            storages.append(
                ExpertStorage(
                    layer_id=entry.layer_id,
                    expert_id=entry.expert_id,
                    activation_frequency=entry.activation_frequency,
                    gate_up_tensor=gate_up,
                    down_tensor=down,
                )
            )
        return cls(storages)

    def __len__(self) -> int:
        return len(self._storages)

    def __iter__(self) -> Iterator[ExpertStorage]:
        return iter(self._storages.values())

    def __getitem__(self, key: ExpertKey) -> ExpertStorage:
        try:
            return self._storages[key]
        except KeyError as exc:
            raise KeyError(f"No storage metadata for expert {key}") from exc

    def create_and_register_regions(self, block_nbytes: int, allocator: UnifiedKVAllocator) -> None:
        for storage in self._storages.values():
            for region in storage.create_temporary_regions(block_nbytes):
                allocator.register_temporary_region(region)
