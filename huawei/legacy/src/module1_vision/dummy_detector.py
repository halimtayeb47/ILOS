# src/module1_vision/dummy_detector.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DetectedTruck:
    id: str
    x: int
    y: int


@dataclass
class DetectedObstacle:
    id: str
    x: int
    y: int


@dataclass
class DetectionFrame:
    trucks: List[DetectedTruck]
    obstacles: List[DetectedObstacle]


class DummyVisionDetector:
    """
    Vision stub that moves trucks step-by-step toward their real goals from the scenario.

    - On each call to get_frame, each truck moves at most 1 cell (4-neighborhood)
      in the direction that reduces its Manhattan distance to the goal.
    - Obstacles are static (same as in the simulator scenario).
    """

    def __init__(
            self,
            width: int = 8,
            height: int = 8,
            truck_starts: Dict[str, Tuple[int, int]] | None = None,
            truck_goals: Dict[str, Tuple[int, int]] | None = None,
            obstacles: List[Tuple[int, int]] | None = None,
    ):
        self.width = width
        self.height = height
        self._step = 0

        # Initialize trucks and goals from scenario data
        self.truck_positions: Dict[str, Tuple[int, int]] = truck_starts or {}
        self.truck_goals: Dict[str, Tuple[int, int]] = truck_goals or {}

        # Static obstacles
        self.obstacle_positions: List[Tuple[int, int]] = obstacles or []

    def reset(self):
        self._step = 0
        # Positions should be reinitialized by the caller if needed

    def _move_toward_goal(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int]:
        """
        Move one step (up/down/left/right) that reduces Manhattan distance to the goal.
        If already at the goal, stay.
        """
        x, y = pos
        gx, gy = goal

        if (x, y) == (gx, gy):
            return pos

        # Candidate moves: stay, up, down, left, right
        candidates = [
            (x, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y),
            (x + 1, y),
        ]

        def in_bounds(p: Tuple[int, int]) -> bool:
            px, py = p
            return 0 <= px < self.width and 0 <= py < self.height

        best_pos = (x, y)
        best_dist = abs(x - gx) + abs(y - gy)

        for p in candidates:
            if not in_bounds(p):
                continue
            # Avoid obstacles (basic check)
            if p in self.obstacle_positions:
                continue
            d = abs(p[0] - gx) + abs(p[1] - gy)
            if d < best_dist:
                best_dist = d
                best_pos = p

        return best_pos

    def get_frame(self) -> DetectionFrame:
        """
        Produce a frame where each truck moves one step closer to its goal,
        ignoring other trucks (no collision logic here, RL agent handles coordination).
        """
        self._step += 1

        new_positions: Dict[str, Tuple[int, int]] = {}
        trucks: List[DetectedTruck] = []

        for tid, pos in self.truck_positions.items():
            goal = self.truck_goals.get(tid, pos)
            new_pos = self._move_toward_goal(pos, goal)
            new_positions[tid] = new_pos
            trucks.append(DetectedTruck(id=tid, x=new_pos[0], y=new_pos[1]))

        # Update internal positions
        self.truck_positions = new_positions

        obstacles = [
            DetectedObstacle(id=f"B{i}", x=ox, y=oy)
            for i, (ox, oy) in enumerate(self.obstacle_positions)
        ]

        return DetectionFrame(trucks=trucks, obstacles=obstacles)

    def to_act_payload(self, frame: DetectionFrame, goals: Dict[str, Tuple[int, int]]) -> dict:
        """
        Convert DetectionFrame to /act payload with rebuild_state=True.
        """
        truck_positions = {
            t.id: {"x": t.x, "y": t.y} for t in frame.trucks
        }
        truck_goals = {
            tid: {"x": g[0], "y": g[1]} for tid, g in goals.items()
        }
        obstacles = [{"x": o.x, "y": o.y} for o in frame.obstacles]

        return {
            "truck_positions": truck_positions,
            "truck_goals": truck_goals,
            "obstacles": obstacles,
            "rebuild_state": True,
        }
