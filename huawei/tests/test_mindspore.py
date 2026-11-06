import sys
from pathlib import Path
import pytest
import numpy as np
pytest.importorskip('mindspore')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mindspore_backend.agent import Agent


def test_training_and_checkpoint_roundtrip(tmp_path):
    agent = Agent(4, 5, hidden_dim=8)
    batch = [(np.zeros(4, dtype=np.float32), 0, 1., np.ones(4, dtype=np.float32), False)] * 8
    assert np.isfinite(agent.update(batch))
    path = agent.save(tmp_path / 'roundtrip.ckpt')
    restored = Agent.load(path)
    assert agent.act(np.zeros(4)) == restored.act(np.zeros(4))
