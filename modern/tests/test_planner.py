import pytest
from ilos.planner import Scenario, Vehicle, PlanningError, Constraint, _search_one, first_conflict, plan


def test_head_on_swap_rejected():
    assert first_conflict({"A": ((0,0),(1,0)), "B": ((1,0),(0,0))}) is not None


def test_intersection_routes_avoid_collision():
    s = Scenario(4, 4, (Vehicle("A", (0,1),(3,1)), Vehicle("B",(1,0),(1,3))))
    r = plan(s)
    r.validate()
    assert first_conflict(r.paths) is None
    assert r.makespan >= 3


def test_parked_vehicle_is_permanent_obstacle():
    s = Scenario(4, 2, (Vehicle("A",(0,0),(1,0)), Vehicle("B",(3,0),(0,0))))
    r = plan(s)
    r.validate()
    assert r.paths["B"][-1] == (0,0)


def test_blocked_goal():
    with pytest.raises(PlanningError):
        Scenario(2,2,(Vehicle("A",(0,0),(1,1)),), obstacles=frozenset({(1,1)}))


def test_unreachable_scenario_raises():
    s = Scenario(3,1,(Vehicle("A",(0,0),(2,0)),),obstacles=frozenset({(1,0)}))
    with pytest.raises(PlanningError):
        plan(s)


def test_congestion_deters_shortcut():
    s = Scenario(3,3,(Vehicle("A",(0,1),(2,1)),),congestion={(1,1):10})
    p = plan(s).paths["A"]
    assert (1,1) not in p


def test_reject_shared_goals():
    with pytest.raises(PlanningError):
        Scenario(3,3,(Vehicle("A",(0,0),(1,1)),Vehicle("B",(2,2),(1,1))))


def test_goal_constraint_checked_after_arrival():
    s = Scenario(2,2,(Vehicle("A",(0,0),(1,0)),))
    p = _search_one(s,s.vehicles[0],frozenset({Constraint("A",4,vertex=(1,0))}),8,1)
    assert p is not None and len(p)>5


def test_zero_length_path():
    s = Scenario(2,2,(Vehicle("A",(0,0),(0,0)),))
    assert plan(s).paths["A"] == ((0,0),)


def test_horizon_failure():
    s = Scenario(4,4,(Vehicle("A",(0,0),(3,3)),))
    with pytest.raises(PlanningError):
        plan(s,horizon=3)


def test_six_vehicle_included_scenario():
    from pathlib import Path
    from ilos.scenario import load_scenario
    scenario = load_scenario(Path(__file__).resolve().parents[1] / "examples" / "six_vehicles.json")
    result = plan(scenario)
    result.validate()
    assert len(result.paths) == 6
    assert result.makespan == 19
    assert result.total_steps == 100


def test_cli_and_offline_html(tmp_path):
    from pathlib import Path
    import json
    from ilos.cli import main
    scenario = Path(__file__).resolve().parents[1] / "examples" / "six_vehicles.json"
    out, html = tmp_path / "routes.json", tmp_path / "demo.html"
    assert main([str(scenario), "--output", str(out), "--html", str(html)]) == 0
    assert json.loads(out.read_text())["collision_free"] is True
    page = html.read_text()
    assert "One coordinated fleet." in page
    assert "https://" not in page  # fully offline, no third-party dependencies


def test_escape_untrusted_scenario_name(tmp_path):
    from ilos.visualize import save_html
    s = Scenario(2, 2, (Vehicle("A",(0,0),(1,0)),), name="</script><script>alert(1)</script>")
    file = tmp_path / "safe.html"
    save_html(plan(s), file)
    assert "</script><script>alert(1)" not in file.read_text()


def test_parked_agent_cannot_move_to_resolve_conflict():
    s = Scenario(3, 2, (Vehicle('A', (1, 0), (1, 0)), Vehicle('B', (0, 0), (2, 0))))
    result = plan(s)
    assert result.paths['A'] == ((1, 0),)
    assert (1, 1) in result.paths['B']


def test_goal_constraint_delays_first_arrival():
    s = Scenario(2, 2, (Vehicle('A', (0, 0), (1, 0)),))
    path = _search_one(s, s.vehicles[0], frozenset({Constraint('A', 4, vertex=(1, 0))}), 8, 1)
    assert path.index((1, 0)) >= 5
