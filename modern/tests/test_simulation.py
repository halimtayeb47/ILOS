import pytest
from ilos.planner import Scenario, Vehicle
from ilos.simulation import FleetSimulator
from ilos.scenario import from_dict


def test_head_on_moves_are_blocked():
    s = Scenario(2, 2, (Vehicle('A', (0, 0), (1, 0)), Vehicle('B', (1, 0), (0, 0))))
    env = FleetSimulator(s)
    _, _, done, _, info = env.step([4, 3])
    assert env.positions == [(0, 0), (1, 0)]
    assert info['conflict_proposals'] == 1
    assert not done


def test_collision_blocking_cascades():
    s = Scenario(4, 2, (Vehicle('A', (0, 0), (3, 0)), Vehicle('B', (1, 0), (2, 0)),
                        Vehicle('C', (2, 0), (1, 0))))
    env = FleetSimulator(s)
    env.step([4, 4, 3])
    assert env.positions == [(0, 0), (1, 0), (2, 0)]


def test_timeout_never_counts_as_arrival():
    env = FleetSimulator(Scenario(3, 1, (Vehicle('A', (0, 0), (2, 0)),)), max_steps=1)
    _, _, terminated, truncated, info = env.step([0])
    assert truncated and not terminated and not info['success']
    with pytest.raises(RuntimeError):
        env.step([4])


def test_goals_are_occupied_after_arrival():
    env = FleetSimulator(Scenario(3, 2, (Vehicle('A', (1, 0), (1, 0)), Vehicle('B', (0, 0), (2, 0)))))
    env.step([4, 4])
    assert env.positions == [(1, 0), (0, 0)]


def test_success_and_observation_dimensions():
    env = FleetSimulator(Scenario(2, 1, (Vehicle('A', (0, 0), (1, 0)),)))
    obs, reward, done, timeout, info = env.step([4])
    assert done and not timeout and info['success']
    assert len(obs) == 5 + 2 * 2
    assert all(0 <= x <= 1 for x in obs)


@pytest.mark.parametrize('width', [True, 2.3, '2', 0])
def test_strict_input_dimensions(width):
    with pytest.raises(ValueError):
        from_dict({'width': width, 'height': 2, 'vehicles': [{'id': 'A', 'start': [0, 0], 'goal': [1, 1]}]})
