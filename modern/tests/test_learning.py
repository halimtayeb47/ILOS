from pathlib import Path
import pytest
pytest.importorskip('stable_baselines3')
from stable_baselines3.common.env_checker import check_env
from ilos.learning import FleetEnv, train, load_model, run_episode
from ilos.planner import Scenario, Vehicle, PlanningError


def test_gym_contract():
    check_env(FleetEnv(Scenario(2, 2, (Vehicle('A', (0, 0), (1, 1)),))), warn=True)


def test_real_training_checkpoint_roundtrip(tmp_path):
    s = Scenario(2, 1, (Vehicle('A', (0, 0), (1, 0)),))
    path = train(s, tmp_path / 'test.zip', steps=256, seed=7, max_steps=5)
    model, env = load_model(s, path)
    paths, report = run_episode(model, env)
    assert 1 <= report['steps'] <= 5
    assert paths['A'][0] == (0, 0)
    other = Scenario(2, 1, (Vehicle('A', (1, 0), (0, 0)),))
    with pytest.raises(PlanningError, match='different scenario'):
        load_model(other, path)
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(PlanningError, match='checksum'):
        load_model(s, path)
