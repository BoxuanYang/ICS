from types import SimpleNamespace

import torch
from minisgl.engine.graph import GraphRunner


class _Graph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class _Buffer:
    def __init__(self) -> None:
        self.logits = torch.ones((1, 2))

    def copy_from(self, batch) -> None:
        pass


def test_replay_switches_between_pure_and_hybrid_graphs(monkeypatch) -> None:
    import minisgl.engine.graph as graph_module

    monkeypatch.setattr(graph_module.logger, "info_rank0", lambda *args: None)
    state = {"hybrid": False}
    pure, hybrid = _Graph(), _Graph()
    runner = GraphRunner.__new__(GraphRunner)
    runner.max_graph_bs = 1
    runner.buffer = _Buffer()
    runner.graph_map = {1: pure}
    runner.hybrid_graph_map = {1: hybrid}
    runner.use_hybrid_graph = lambda: state["hybrid"]
    runner._logged_replay_variants = set()
    runner.attn_backend = SimpleNamespace(prepare_for_replay=lambda batch: None)
    batch = SimpleNamespace(is_decode=True, size=1, padded_size=1)

    torch.testing.assert_close(runner.replay(batch), torch.ones((1, 2)))
    assert (pure.replays, hybrid.replays) == (1, 0)

    state["hybrid"] = True
    runner.replay(batch)
    assert (pure.replays, hybrid.replays) == (1, 1)
    assert runner._logged_replay_variants == {"pure-gpu", "hybrid"}
