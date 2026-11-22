from __future__ import annotations

from typing import NamedTuple


class WeightAllocation(NamedTuple):
    parameters: int
    nbytes: int


class ModelWeightAllocation(NamedTuple):
    moe: WeightAllocation
    other: WeightAllocation


def model_weight_allocation(model) -> ModelWeightAllocation:
    """Split rank-local model weights into expert matrices and everything else."""
    moe_parameters = 0
    moe_nbytes = 0
    other_parameters = 0
    other_nbytes = 0
    for name, tensor in model.state_dict().items():
        parameters = tensor.numel()
        nbytes = parameters * tensor.element_size()
        if ".mlp.experts." in name:
            moe_parameters += parameters
            moe_nbytes += nbytes
        else:
            other_parameters += parameters
            other_nbytes += nbytes
    return ModelWeightAllocation(
        moe=WeightAllocation(moe_parameters, moe_nbytes),
        other=WeightAllocation(other_parameters, other_nbytes),
    )
