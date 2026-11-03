from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend
from .memory import ExpertMemoryManager
from .offload import (
    EvictionPlan,
    ExpertEvictionProfile,
    ExpertKey,
    ExpertOffloadController,
    ExpertProfileEntry,
    ExpertResidencyState,
    ExpertResidencyTable,
    KVPressureSnapshot,
)
from .storage import ExpertStorage, ExpertStorageCatalog

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "EvictionPlan",
    "ExpertEvictionProfile",
    "ExpertKey",
    "ExpertMemoryManager",
    "ExpertOffloadController",
    "ExpertProfileEntry",
    "ExpertResidencyState",
    "ExpertResidencyTable",
    "ExpertStorage",
    "ExpertStorageCatalog",
    "KVPressureSnapshot",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
