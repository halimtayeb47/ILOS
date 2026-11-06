"""Deterministic fleet dynamics shared by training and evaluation.

Moves are simultaneous. Conflicting proposals are stopped, including cascades
where stopping one vehicle would otherwise let another drive into it.
"""
from __future__ import annotations

from .planner import Scenario

MOVES = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))


class FleetSimulator:
    def __init__(self, scenario: Scenario, max_steps: int = 120):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.scenario = scenario
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.positions = [v.start for v in self.scenario.vehicles]
        self.goals = [v.goal for v in self.scenario.vehicles]
        self.done = [p == g for p, g in zip(self.positions, self.goals)]
        self.steps = 0
        self.finished = False
        return self.observation()

    def observation(self):
        s = self.scenario
        out = []
        for p, g, done in zip(self.positions, self.goals, self.done):
            out.extend((p[0] / max(s.width - 1, 1), p[1] / max(s.height - 1, 1),
                        g[0] / max(s.width - 1, 1), g[1] / max(s.height - 1, 1), float(done)))
        out.extend(float((x, y) in s.obstacles) for y in range(s.height) for x in range(s.width))
        out.extend(s.congestion.get((x, y), 0) / (1 + s.congestion.get((x, y), 0))
                   for y in range(s.height) for x in range(s.width))
        return out

    def step(self, actions):
        if self.finished:
            raise RuntimeError("Reset the simulation after termination or timeout")
        if len(actions) != len(self.positions) or any(type(a) is not int or not 0 <= a < 5 for a in actions):
            raise ValueError("Supply one integer action (0..4) per vehicle")
        old = self.positions[:]
        proposed = []
        blocked = set()
        for i, (p, action) in enumerate(zip(old, actions)):
            dx, dy = MOVES[action]
            nxt = p if self.done[i] else (p[0] + dx, p[1] + dy)
            if not self.scenario.valid(nxt):
                blocked.add(i)
                nxt = p
            proposed.append(nxt)
        conflicts = set()
        while True:
            stop = set()
            for i in range(len(old)):
                for j in range(i + 1, len(old)):
                    vertex = proposed[i] == proposed[j]
                    swap = proposed[i] == old[j] and proposed[j] == old[i]
                    if vertex or swap:
                        conflicts.add((i, j))
                        stop.update((i, j))
            moving = {i for i in stop if proposed[i] != old[i]}
            if not moving:
                break
            blocked.update(moving)
            for i in moving:
                proposed[i] = old[i]
        reward = 0.0
        for i, (p, q, goal) in enumerate(zip(old, proposed, self.goals)):
            if self.done[i]:
                continue
            distance = lambda v: abs(v[0] - goal[0]) + abs(v[1] - goal[1])
            reward += -0.05 + 0.25 * (distance(p) - distance(q))
            reward -= 0.05 * self.scenario.congestion.get(q, 0)
            if i in blocked:
                reward -= 1.0
            if q == goal:
                self.done[i] = True
                reward += 10.0
        reward -= len(conflicts)
        self.positions = proposed
        self.steps += 1
        terminated = all(self.done)
        truncated = self.steps >= self.max_steps and not terminated
        self.finished = terminated or truncated
        return self.observation(), reward, terminated, truncated, {
            "success": terminated, "steps": self.steps,
            "conflict_proposals": len(conflicts), "blocked_moves": len(blocked),
            "arrived": sum(self.done), "positions": list(proposed),
        }
