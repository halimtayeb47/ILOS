# src/simulator/renderer.py

from __future__ import annotations
from typing import Optional

from .environment import MultiTruckGridEnv
from .scenario_loader import load_scenario


class SimpleTextRenderer:
    """
    Simple text-mode renderer around MultiTruckGridEnv.
    """

    def __init__(self, env: MultiTruckGridEnv):
        self.env = env

    def render(self):
        return self.env.render()


def make_env_from_yaml(path: str, render_mode: Optional[str] = "human") -> MultiTruckGridEnv:
    scenario = load_scenario(path)
    env = MultiTruckGridEnv(scenario=scenario, render_mode=render_mode)
    return env
