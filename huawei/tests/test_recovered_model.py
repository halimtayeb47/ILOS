import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'legacy'))
pytest.importorskip('torch')
pytest.importorskip('yaml')
from src.module2_rl.rl_agent import DQNAgent
from src.simulator.environment import MultiTruckGridEnv
from src.simulator.scenario_loader import load_scenario


def test_recovered_checkpoint_matches_scenario():
    env = MultiTruckGridEnv(load_scenario(str(ROOT / 'legacy/data/scenarios/example_scenario1.yaml')))
    obs, _ = env.reset()
    agent = DQNAgent.load(str(ROOT / 'checkpoints/dqn_final.pt'), device='cpu')
    assert agent.cfg.state_dim == len(obs)
    assert agent.cfg.action_dim == env.action_space.n
    assert env.action_space.contains(agent.select_action(obs, evaluate=True))


def test_legacy_timeout_is_not_success():
    scenario = load_scenario(str(ROOT / 'legacy/data/scenarios/example_scenario1.yaml'))
    scenario.world.max_steps = 1
    env = MultiTruckGridEnv(scenario)
    env.reset()
    _, _, done, truncated, info = env.step(0)
    assert truncated and not done and not info['success']
