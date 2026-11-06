# src/module2_rl/evaluate.py

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from src.simulator.scenario_loader import load_scenario
from src.simulator.environment import MultiTruckGridEnv
from .rl_agent import DQNAgent


def make_env(scenario_path: str, render_mode: Optional[str] = "human") -> MultiTruckGridEnv:
    scenario = load_scenario(scenario_path)
    env = MultiTruckGridEnv(scenario=scenario, render_mode=render_mode)
    return env


def evaluate(
        model_path: str,
        scenario_path: str,
        episodes: int = 5,
        device: str = "cuda",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    agent = DQNAgent.load(model_path, device=device)
    env = make_env(scenario_path, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

        print(
            f"[EVAL] Episode {ep} finished in {steps} steps, total_reward={total_reward:.2f}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN on MultiTruckGridEnv.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to trained model (.pt).",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="data/scenarios/example_scenario1.yaml",
        help="Path to scenario YAML.",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        scenario_path=args.scenario,
        episodes=args.episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
