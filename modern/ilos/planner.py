"""Collision-aware multi-agent grid planning with conflict-based search (CBS).

Every transition has unit duration. Obstacles are static, positions are integer
(x, y) coordinates, and vehicles remain at their goals after arrival.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import count
from math import inf
from time import monotonic
from typing import Iterable

Cell = tuple[int, int]
Path = tuple[Cell, ...]


class PlanningError(ValueError):
    """Invalid or unsolvable planning problem, or exhausted search budget."""


@dataclass(frozen=True)
class Vehicle:
    id: str
    start: Cell
    goal: Cell


@dataclass
class Scenario:
    width: int
    height: int
    vehicles: tuple[Vehicle, ...]
    obstacles: frozenset[Cell] = frozenset()
    congestion: dict[Cell, float] = field(default_factory=dict)
    name: str = "ILOS scenario"

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int or self.width < 1 or self.height < 1:
            raise PlanningError("Map width and height must be positive")
        if not self.vehicles:
            raise PlanningError("At least one vehicle is required")
        ids = [v.id for v in self.vehicles]
        if len(ids) != len(set(ids)) or any(not s for s in ids):
            raise PlanningError("Vehicle IDs must be unique and nonempty")
        starts = [v.start for v in self.vehicles]
        goals = [v.goal for v in self.vehicles]
        if len(set(starts)) != len(starts):
            raise PlanningError("Vehicles may not share a starting cell")
        if len(set(goals)) != len(goals):
            raise PlanningError("Vehicles may not share a destination cell (goals remain occupied)")
        for p in self.obstacles:
            if not self.in_bounds(p):
                raise PlanningError(f"Obstacle out of bounds: {p}")
        for p in [*starts, *goals, *self.congestion]:
            if not self.valid(p):
                raise PlanningError(f"Invalid/out-of-bounds/blocked coordinate: {p}")
        if any(weight < 0 or weight == inf or weight != weight for weight in self.congestion.values()):
            raise PlanningError("Congestion weights must be finite, nonnegative numbers")

    def in_bounds(self, p: Cell) -> bool:
        return len(p) == 2 and all(type(x) is int for x in p) and 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def valid(self, p: Cell) -> bool:
        return self.in_bounds(p) and p not in self.obstacles

    def neighbors(self, p: Cell) -> Iterable[Cell]:
        x, y = p
        for next_p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), p):
            if self.valid(next_p):
                yield next_p


# Constraints bind a vehicle to a forbidden vertex at time t, or edge at t.
# A forbidden edge is the movement from u at t-1 to v at t.
@dataclass(frozen=True)
class Constraint:
    agent: str
    time: int
    vertex: Cell | None = None
    edge: tuple[Cell, Cell] | None = None


def at(path: Path, time: int) -> Cell:
    return path[min(time, len(path) - 1)]


def first_conflict(paths: dict[str, Path]) -> tuple[str, str, int, Cell | None, tuple[Cell, Cell] | None, tuple[Cell, Cell] | None] | None:
    """Detect first vertex conflict or head-on edge swap, including parked goals."""
    agents = sorted(paths)
    until = max(len(p) for p in paths.values())
    for t in range(until):
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                pa, pb = paths[a], paths[b]
                if at(pa, t) == at(pb, t):
                    return a, b, t, at(pa, t), None, None
                if t and at(pa, t - 1) == at(pb, t) and at(pb, t - 1) == at(pa, t):
                    return a, b, t, None, (at(pa, t - 1), at(pa, t)), (at(pb, t - 1), at(pb, t))
    return None


def path_cost(path: Path, scenario: Scenario, congestion_factor: float = 1.0) -> float:
    return (len(path) - 1) + sum(congestion_factor * scenario.congestion.get(p, 0) for p in path[1:])


def _can_finish(goal: Cell, time: int, rules: frozenset[Constraint], horizon: int) -> bool:
    # A parked vehicle persists through the planning horizon; later constraints matter.
    return all(not (
        rule.time >= time
        and ((rule.vertex == goal) or (rule.edge == (goal, goal)))
    ) for rule in rules)


def _search_one(scenario: Scenario, vehicle: Vehicle, rules: frozenset[Constraint],
                horizon: int, congestion_factor: float, deadline: float = inf) -> Path | None:
    relevant = frozenset(c for c in rules if c.agent == vehicle.id)
    if Constraint(vehicle.id, 0, vertex=vehicle.start) in relevant:
        return None
    if vehicle.start == vehicle.goal:
        return (vehicle.start,) if _can_finish(vehicle.goal, 0, relevant, horizon) else None
    counter = count()
    # h = Manhattan distance; congestion penalties are nonnegative.
    h = lambda p: abs(p[0] - vehicle.goal[0]) + abs(p[1] - vehicle.goal[1])
    queue: list[tuple[float, float, int, int, Cell]] = [(h(vehicle.start), 0, next(counter), 0, vehicle.start)]
    best = {(vehicle.start, 0): 0.0}
    parent: dict[tuple[Cell, int], tuple[Cell, int]] = {}
    while queue:
        if monotonic() > deadline:
            raise PlanningError("Planning time limit exceeded")
        _, g, _, t, p = heapq.heappop(queue)
        if g > best.get((p, t), inf) + 1e-9:
            continue
        if p == vehicle.goal and _can_finish(p, t, relevant, horizon):
            out: list[Cell] = [p]
            cursor = (p, t)
            while cursor in parent:
                cursor = parent[cursor]
                out.append(cursor[0])
            return tuple(reversed(out))
        if t >= horizon:
            continue
        for nxt in scenario.neighbors(p):
            nt = t + 1
            if Constraint(vehicle.id, nt, vertex=nxt) in relevant or Constraint(vehicle.id, nt, edge=(p, nxt)) in relevant:
                continue
            if nxt == vehicle.goal and not _can_finish(nxt, nt, relevant, horizon):
                continue
            ng = g + 1 + congestion_factor * scenario.congestion.get(nxt, 0)
            if ng + 1e-9 < best.get((nxt, nt), inf):
                best[(nxt, nt)] = ng
                parent[(nxt, nt)] = (p, t)
                heapq.heappush(queue, (ng + h(nxt), ng, next(counter), nt, nxt))
    return None


@dataclass
class Result:
    paths: dict[str, Path]
    algorithm: str
    expanded_nodes: int
    scenario: Scenario
    congestion_factor: float = 1.0

    @property
    def makespan(self) -> int:
        return max(len(p) - 1 for p in self.paths.values())

    @property
    def total_steps(self) -> int:
        return sum(len(p) - 1 for p in self.paths.values())

    @property
    def weighted_cost(self) -> float:
        return round(sum(path_cost(path, self.scenario, self.congestion_factor) for path in self.paths.values()), 5)

    def validate(self) -> None:
        if set(self.paths) != {v.id for v in self.scenario.vehicles}:
            raise PlanningError("Missing or unknown vehicle routes")
        for v in self.scenario.vehicles:
            path = self.paths[v.id]
            if not path or path[0] != v.start or path[-1] != v.goal:
                raise PlanningError(f"Incorrect start or goal: {v.id}")
            first_arrival = path.index(v.goal)
            if any(p != v.goal for p in path[first_arrival:]):
                raise PlanningError(f"Vehicle leaves its goal after arrival: {v.id}")
            if any(not self.scenario.valid(p) for p in path):
                raise PlanningError(f"Blocked or out-of-bounds route: {v.id}")
            if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for a, b in zip(path, path[1:])):
                raise PlanningError(f"Nonadjacent transition: {v.id}")
        conflict = first_conflict(self.paths)
        if conflict:
            raise PlanningError(f"Collision detected: {conflict}")


def plan(scenario: Scenario, *, horizon: int | None = None,
         max_nodes: int = 12000, congestion_factor: float = 1.0, timeout: float = 30.0) -> Result:
    """Bounded CBS minimizing summed weighted travel cost within the horizon.

    Raising PlanningError on budget exhaustion is deliberate: never return a
    possibly-colliding schedule as a 'successful' plan.
    """
    if not 0 < timeout < inf:
        raise PlanningError("timeout must be finite and positive")
    deadline = monotonic() + timeout
    if congestion_factor < 0 or congestion_factor == inf or congestion_factor != congestion_factor:
        raise PlanningError("Congestion factor must be finite and nonnegative")
    if type(max_nodes) is not int or max_nodes <= 0:
        raise PlanningError("max_nodes must be positive")
    if horizon is None:
        horizon = max(24, scenario.width * scenario.height * 2 + len(scenario.vehicles) * 4)
    if type(horizon) is not int or horizon < 0:
        raise PlanningError("horizon must be nonnegative")
    empty_rules: frozenset[Constraint] = frozenset()
    root_paths: dict[str, Path] = {}
    for v in scenario.vehicles:
        p = _search_one(scenario, v, empty_rules, horizon, congestion_factor, deadline)
        if p is None:
            raise PlanningError(f"No route exists within horizon for {v.id}")
        root_paths[v.id] = p
    pushid = count()
    cost = lambda paths: sum(path_cost(p, scenario, congestion_factor) for p in paths.values())
    queue: list[tuple[float, int, frozenset[Constraint], dict[str, Path]]] = [
        (cost(root_paths), next(pushid), empty_rules, root_paths)
    ]
    seen = {empty_rules}
    vehicles_by_id = {v.id: v for v in scenario.vehicles}
    expanded = 0
    while queue and expanded < max_nodes:
        if monotonic() > deadline:
            raise PlanningError("Planning time limit exceeded")
        _, _, rules, paths = heapq.heappop(queue)
        expanded += 1
        conflict = first_conflict(paths)
        if conflict is None:
            result = Result(paths, "Conflict-Based Search (CBS)", expanded, scenario, congestion_factor)
            result.validate()
            return result
        a, b, t, vertex, ea, eb = conflict
        for agent, edge in ((a, ea), (b, eb)):
            rule = Constraint(agent, t, vertex=vertex, edge=edge)
            child_rules = rules | {rule}
            if child_rules in seen:
                continue
            seen.add(child_rules)
            new_path = _search_one(scenario, vehicles_by_id[agent], child_rules, horizon, congestion_factor, deadline)
            if new_path is None:
                continue
            child_paths = dict(paths)
            child_paths[agent] = new_path
            heapq.heappush(queue, (cost(child_paths), next(pushid), child_rules, child_paths))
    raise PlanningError(f"No collision-free solution within horizon={horizon}, max_nodes={max_nodes}; try raising limits or revising the scenario")
