# src/simulator/scenario_loader.py

from __future__ import annotations
from typing import Dict, Any, List
import os
import yaml

from .entities import WorldConfig, Obstacle, Scenario


def load_scenario(path: str) -> Scenario:
    """
    Load a YAML scenario file and return a Scenario object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    name = data.get("name", os.path.splitext(os.path.basename(path))[0])

    world_cfg = data["world"]
    world = WorldConfig(
        width=world_cfg["width"],
        height=world_cfg["height"],
        n_trucks=world_cfg["n_trucks"],
        max_steps=world_cfg.get("max_steps", 200),
        collision_penalty=world_cfg.get("collision_penalty", -10.0),
        step_penalty=world_cfg.get("step_penalty", -0.1),
        goal_reward=world_cfg.get("goal_reward", 10.0),
        conflict_penalty=world_cfg.get("conflict_penalty", -5.0),
        out_of_bounds_penalty=world_cfg.get("out_of_bounds_penalty", -5.0),
    )

    truck_starts: Dict[str, Any] = data["truck_starts"]
    truck_goals: Dict[str, Any] = data["truck_goals"]

    obstacles_raw: List[Dict[str, Any]] = data.get("obstacles", [])
    obstacles = [
        Obstacle(
            id=str(o.get("id", f"obs_{i}")),
            position=(int(o["x"]), int(o["y"])),
        )
        for i, o in enumerate(obstacles_raw)
    ]

    return Scenario(
        name=name,
        world=world,
        truck_starts={
            str(k): (int(v["x"]), int(v["y"])) for k, v in truck_starts.items()
        },
        truck_goals={
            str(k): (int(v["x"]), int(v["y"])) for k, v in truck_goals.items()
        },
        obstacles=obstacles,
    )
