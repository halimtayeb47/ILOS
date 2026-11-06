# src/simulator/environment.py

from __future__ import annotations

import itertools
from typing import Dict, Tuple, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .entities import Truck, Scenario, WorldConfig, Position


Action = int


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class MultiTruckGridEnv(gym.Env):
    """
    Multi-truck grid environment for logistics routing.

    - Centralized control: one agent controls all trucks.
    - Discrete 2D grid with obstacles.
    - Reward shaping: distance-based towards goals + penalties for conflicts.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
            self,
            scenario: Scenario,
            render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.scenario = scenario
        self.world: WorldConfig = scenario.world
        self.render_mode = render_mode

        self.n_trucks = self.world.n_trucks
        self.truck_ids: List[str] = sorted(list(self.scenario.truck_starts.keys()))
        if len(self.truck_ids) != self.n_trucks:
            raise ValueError(
                f"Scenario n_trucks={self.n_trucks} but truck_starts has {len(self.truck_ids)} entries"
            )

        self.obstacles = {o.position for o in self.scenario.obstacles}
        self.trucks: Dict[str, Truck] = {}

        # Each truck: 5 actions (stay, up, down, left, right)
        self.n_actions_per_truck = 5
        self.action_space = spaces.Discrete(self.n_actions_per_truck ** self.n_trucks)

        # Observation:
        # For each truck: (x, y, goal_x, goal_y, done_flag)
        # plus flattened obstacle grid (0/1 for each cell)
        obs_truck_dim = 5
        self.obs_truck_dim = obs_truck_dim
        self.grid_size = self.world.width * self.world.height
        self.obstacles_dim = self.grid_size

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0] * (self.n_trucks * obs_truck_dim + self.obstacles_dim),
                dtype=np.float32,
                ),
            high=np.array(
                [1.0] * (self.n_trucks * obs_truck_dim + self.obstacles_dim),
                dtype=np.float32,
                ),
            dtype=np.float32,
        )

        self._current_step: int = 0
        # لتخزين المسافات السابقة للأهداف لكل شاحنة (للـ shaping)
        self._prev_distances: Dict[str, float] = {}

    # -------------- Gym API --------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._current_step = 0
        self.trucks.clear()
        self._prev_distances.clear()

        # Initialize trucks
        for idx, truck_id in enumerate(self.truck_ids):
            start_pos = self.scenario.truck_starts[truck_id]
            goal_pos = self.scenario.truck_goals[truck_id]
            truck = Truck(
                id=truck_id,
                position=start_pos,
                goal_id=truck_id,
                max_steps=self.world.max_steps,
                done=(start_pos == goal_pos),
            )
            self.trucks[truck_id] = truck

            # initial distance
            self._prev_distances[truck_id] = float(manhattan(start_pos, goal_pos))

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: Action):
        if not self.action_space.contains(action):
            raise ValueError("Joint action is outside the action space")
        self._current_step += 1

        truck_actions = self._decode_action(action)

        # tentative new positions
        new_positions: Dict[str, Position] = {}
        for truck_id, truck in self.trucks.items():
            if truck.done:
                new_positions[truck_id] = truck.position
                continue

            move = truck_actions[truck_id]
            new_positions[truck_id] = self._apply_move(truck.position, move)

        # bounds + obstacles
        for truck_id, new_pos in list(new_positions.items()):
            if not self._in_bounds(new_pos) or new_pos in self.obstacles:
                # invalid move -> stay put + later we add penalty
                new_positions[truck_id] = self.trucks[truck_id].position

        # conflicts (same cell)
        position_counts: Dict[Position, int] = {}
        for pos in new_positions.values():
            position_counts[pos] = position_counts.get(pos, 0) + 1

        conflicts_positions = {pos for pos, c in position_counts.items() if c > 1}

        total_reward = 0.0
        terminated = False
        truncated = False

        # rewards per truck (shaped)
        for truck_id, truck in self.trucks.items():
            old_pos = truck.position
            new_pos = new_positions[truck_id]
            goal_pos = self.scenario.truck_goals[truck_id]

            # base step penalty صغير
            r = self.world.step_penalty

            # distance-based shaping: مكافأة صغيرة عند الاقتراب من الهدف
            prev_d = self._prev_distances[truck_id]
            curr_d = float(manhattan(new_pos, goal_pos))
            distance_diff = prev_d - curr_d  # >0 إذا اقترب من الهدف

            # نضيف مكافأة متناسبة مع distance_diff
            r += 0.2 * distance_diff  # معامل يمكن تعديله

            # conflict penalty لو شاحنتين بنفس الخلية
            if new_pos in conflicts_positions:
                r += self.world.conflict_penalty

            # goal check
            if (not truck.done) and (new_pos == goal_pos):
                r += self.world.goal_reward
                truck.done = True
                curr_d = 0.0

            # out-of-bounds / obstacle invalid: تم التعامل معها بتثبيت الموضع،
            # لكن نضيف penalty إضافي لو المحاولة كانت خروجًا أو إلى عائق
            if new_pos == old_pos and truck_actions[truck_id] != 0:
                # محاولة تحرك لكن انتهت بنفس المكان -> غالبًا اصطدام بحد/عائق
                r += self.world.out_of_bounds_penalty

            truck.position = new_pos
            truck.step_counter()
            self._prev_distances[truck_id] = curr_d

            total_reward += r

        # termination
        if all(truck.done for truck in self.trucks.values()):
            terminated = True
        if self._current_step >= self.world.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            "step": self._current_step,
            "conflicts": len(conflicts_positions),
            "success": all(t.position == self.scenario.truck_goals[k] for k, t in self.trucks.items()),
        }

        return obs, total_reward, terminated, truncated, info

    # -------------- Helpers --------------

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.world.width and 0 <= y < self.world.height

    def _apply_move(self, pos: Position, move: int) -> Position:
        """
        move: 0 = stay, 1 = up, 2 = down, 3 = left, 4 = right
        """
        x, y = pos
        if move == 1:  # up
            return (x, y - 1)
        if move == 2:  # down
            return (x, y + 1)
        if move == 3:  # left
            return (x - 1, y)
        if move == 4:  # right
            return (x + 1, y)
        return pos

    def _decode_action(self, action: Action) -> Dict[str, int]:
        a = int(action)
        base = self.n_actions_per_truck
        per_truck = {}
        for truck_id in self.truck_ids:
            per_truck[truck_id] = a % base
            a //= base
        return per_truck

    def _get_observation(self) -> np.ndarray:
        w = self.world.width
        h = self.world.height

        trucks_features: List[float] = []
        for truck_id in self.truck_ids:
            truck = self.trucks[truck_id]
            goal_pos = self.scenario.truck_goals[truck_id]
            x_norm = truck.position[0] / max(w - 1, 1)
            y_norm = truck.position[1] / max(h - 1, 1)
            gx_norm = goal_pos[0] / max(w - 1, 1)
            gy_norm = goal_pos[1] / max(h - 1, 1)
            done_flag = 1.0 if truck.done else 0.0
            trucks_features.extend([x_norm, y_norm, gx_norm, gy_norm, done_flag])

        obstacle_grid = np.zeros((h, w), dtype=np.float32)
        for (ox, oy) in self.obstacles:
            if 0 <= ox < w and 0 <= oy < h:
                obstacle_grid[oy, ox] = 1.0

        obs = np.concatenate(
            [np.array(trucks_features, dtype=np.float32), obstacle_grid.flatten()]
        )
        return obs

    # -------------- Rendering --------------

    def render(self):
        if self.render_mode is None:
            return

        grid = np.full((self.world.height, self.world.width), ".", dtype=object)

        # Obstacles
        for (ox, oy) in self.obstacles:
            grid[oy, ox] = "#"

        # Trucks
        for idx, truck_id in enumerate(self.truck_ids):
            truck = self.trucks[truck_id]
            x, y = truck.position
            if 0 <= x < self.world.width and 0 <= y < self.world.height:
                grid[y, x] = str(idx)

        # Goals
        for idx, truck_id in enumerate(self.truck_ids):
            gx, gy = self.scenario.truck_goals[truck_id]
            if 0 <= gx < self.world.width and 0 <= gy < self.world.height:
                if isinstance(grid[gy, gx], str) and grid[gy, gx].isdigit():
                    grid[gy, gx] = "T"
                elif grid[gy, gx] == ".":
                    grid[gy, gx] = "G"

        if self.render_mode == "ansi":
            return self._ansi_render(grid)
        elif self.render_mode == "human":
            print(self._ansi_render(grid))

    def _ansi_render(self, grid: np.ndarray) -> str:
        lines = []
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)

    def close(self):
        pass
