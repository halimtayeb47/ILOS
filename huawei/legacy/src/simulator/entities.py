# src/simulator/entities.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional


Position = Tuple[int, int]


@dataclass
class Obstacle:
    id: str
    position: Position


@dataclass
class Goal:
    id: str
    position: Position


@dataclass
class Truck:
    id: str
    position: Position
    goal_id: str
    max_steps: int = 500
    steps_taken: int = 0
    done: bool = False

    def reset(self, position: Position, goal_id: str) -> None:
        self.position = position
        self.goal_id = goal_id
        self.steps_taken = 0
        self.done = False

    def step_counter(self) -> None:
        self.steps_taken += 1


@dataclass
class WorldConfig:
    width: int
    height: int
    n_trucks: int
    max_steps: int
    collision_penalty: float
    step_penalty: float
    goal_reward: float
    conflict_penalty: float
    out_of_bounds_penalty: float


@dataclass
class Scenario:
    name: str
    world: WorldConfig
    truck_starts: Dict[str, Position]
    truck_goals: Dict[str, Position]
    obstacles: List[Obstacle]
