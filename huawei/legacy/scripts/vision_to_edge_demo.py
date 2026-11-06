# scripts/vision_to_edge_demo.py

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple, List

import requests

from src.module1_vision.dummy_detector import DummyVisionDetector
from src.simulator.scenario_loader import load_scenario


def run_vision_edge_loop(
        scenario_path: str,
        base_url: str = "http://127.0.0.1:8000",
        steps: int = 30,
        sleep_sec: float = 0.3,
):
    scenario = load_scenario(scenario_path)

    # Extract starts, goals, obstacles from scenario
    truck_starts: Dict[str, Tuple[int, int]] = {
        tid: (pos[0], pos[1]) for tid, pos in scenario.truck_starts.items()
    }
    truck_goals: Dict[str, Tuple[int, int]] = {
        tid: (pos[0], pos[1]) for tid, pos in scenario.truck_goals.items()
    }
    obstacle_positions: List[Tuple[int, int]] = [o.position for o in scenario.obstacles]

    detector = DummyVisionDetector(
        width=scenario.world.width,
        height=scenario.world.height,
        truck_starts=truck_starts,
        truck_goals=truck_goals,
        obstacles=obstacle_positions,
    )
    detector.reset()

    goals = truck_goals  # already in (x,y) form

    for step in range(steps):
        frame = detector.get_frame()
        payload = detector.to_act_payload(frame, goals)

        r = requests.post(f"{base_url}/act", json=payload, timeout=5.0)
        if r.status_code != 200:
            print(f"[step {step}] /act error: {r.status_code} {r.text}")
            break

        data = r.json()
        action = data["action"]
        per_truck_actions = data["per_truck_actions"]

        print(f"[step {step}]")
        print(f"  trucks: {payload['truck_positions']}")
        print(f"  obstacles: {payload['obstacles']}")
        print(f"  actions: {per_truck_actions} (joint={action})")

        if sleep_sec > 0:
            time.sleep(sleep_sec)


def main():
    parser = argparse.ArgumentParser(description="Vision stub -> Edge RL demo (goal-following).")
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
        help="Edge server base URL.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of steps to run.",
    )
    args = parser.parse_args()

    run_vision_edge_loop(
        scenario_path=args.scenario,
        base_url=args.base_url,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
