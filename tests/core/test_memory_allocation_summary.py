import torch
from minisgl.memory_usage import model_weight_allocation


class _FakeModel:
    def state_dict(self):
        return {
            "model.layers.0.mlp.experts.gate_up_proj": torch.empty(
                3, 4, dtype=torch.bfloat16
            ),
            "model.layers.0.mlp.experts.down_proj": torch.empty(5, dtype=torch.float16),
            "model.layers.0.mlp.shared_experts.down_proj.weight": torch.empty(9, dtype=torch.bfloat16),
            "model.layers.0.mlp.gate.weight": torch.empty(7, dtype=torch.float32),
        }


def test_model_weight_allocation_splits_experts_from_other_weights() -> None:
    allocation = model_weight_allocation(_FakeModel())

    assert allocation.moe.parameters == 26
    assert allocation.moe.nbytes == 52
    assert allocation.other.parameters == 7
    assert allocation.other.nbytes == 28
