# scripts/run_sim_scenarios.py

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from src.simulator.renderer import make_env_from_yaml


def run_episode(
        scenario_path: str,
        max_episodes: int = 1,
        render: bool = True,
        seed: Optional[int] = None,
):
    env = make_env_from_yaml(scenario_path, render_mode="human" if render else None)

    for ep in range(max_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        total_reward = 0.0

        step = 0
        while not (done or truncated):
            # Random joint action for now (to be replaced by DQN)
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()

        print(
            f"Episode {ep} finished in {step} steps, total_reward={total_reward:.2f}, info={info}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run logistics simulator scenarios.")
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="data/scenarios/example_scenario1.yaml",
        help="Path to the YAML scenario file.",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering.",
    )
    args = parser.parse_args()

    scenario_path = args.scenario
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    run_episode(
        scenario_path=scenario_path,
        max_episodes=args.episodes,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
