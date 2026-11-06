# scripts/call_edge_server.py

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import requests

from src.simulator.scenario_loader import load_scenario
from src.simulator.environment import MultiTruckGridEnv


def run_episode_with_remote_agent(
        scenario_path: str,
        base_url: str = "http://127.0.0.1:8000",
        episodes: int = 3,
        sleep_sec: float = 0.1,
):
    scenario = load_scenario(scenario_path)
    env = MultiTruckGridEnv(scenario=scenario, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        print(f"\n=== Episode {ep} ===")

        while not (done or truncated):
            # نرسل الحالة الحالية للسيرفر
            payload = {
                "state": obs.tolist()
            }
            r = requests.post(f"{base_url}/act", json=payload, timeout=5.0)
            if r.status_code != 200:
                raise RuntimeError(f"/act failed: {r.status_code} {r.text}")

            data = r.json()
            action = data["action"]
            per_truck_actions = data["per_truck_actions"]

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            env.render()
            print(
                f"step={steps} reward={reward:.2f} total={total_reward:.2f} "
                f"actions={per_truck_actions}"
            )

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        print(
            f"Episode {ep} finished in {steps} steps, total_reward={total_reward:.2f}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test remote RL agent via FastAPI /act.")
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="data/scenarios/example_scenario1.yaml",
        help="Scenario YAML path.",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of edge server.",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=3,
        help="Number of episodes.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Pause between steps (seconds).",
    )
    args = parser.parse_args()

    run_episode_with_remote_agent(
        scenario_path=args.scenario,
        base_url=args.base_url,
        episodes=args.episodes,
        sleep_sec=args.sleep,
    )


if __name__ == "__main__":
    main()
