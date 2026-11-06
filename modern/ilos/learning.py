"""PPO training, checkpoint metadata, evaluation and validated route export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .planner import PlanningError, Result
from .scenario import load_scenario, scenario_dict
from .simulation import FleetSimulator


class FleetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario, max_steps=120):
        self.sim = FleetSimulator(scenario, max_steps)
        self.action_space = spaces.MultiDiscrete([5] * len(scenario.vehicles))
        self.observation_space = spaces.Box(0, 1, shape=(len(self.sim.observation()),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray(self.sim.reset(), dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action does not match the fleet")
        obs, reward, terminated, truncated, info = self.sim.step([int(a) for a in action])
        return np.asarray(obs, dtype=np.float32), reward, terminated, truncated, info


def fingerprint(scenario):
    data = scenario_dict(scenario)
    data.pop("name", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def train(scenario, output, *, steps=100_000, seed=42, max_steps=120):
    import torch
    if steps < 1:
        raise ValueError("Training steps must be positive")
    torch.set_num_threads(1)
    env = Monitor(FleetEnv(scenario, max_steps))
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", n_steps=256,
                batch_size=64, n_epochs=10, learning_rate=3e-4, ent_coef=0.01,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}}, verbose=0)
    model.learn(total_timesteps=steps)
    target = Path(output).with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target, exclude=["_last_obs", "_last_episode_starts", "_last_original_obs"])
    meta = {"algorithm": "PPO", "schema_version": 1, "seed": seed,
            "requested_steps": steps, "trained_steps": model.num_timesteps,
            "max_episode_steps": max_steps, "scenario_sha256": fingerprint(scenario),
            "vehicle_ids": [v.id for v in scenario.vehicles],
            "checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    env.close()
    return target


def load_model(scenario, checkpoint):
    path = Path(checkpoint)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise PlanningError("PPO checkpoint and its JSON metadata are required; run ilos-train first")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("scenario_sha256") != fingerprint(scenario):
        raise PlanningError("PPO model was trained for a different scenario; retrain for this map and fleet")
    if meta.get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise PlanningError("PPO checkpoint checksum does not match its metadata")
    # Only load checkpoints you created or explicitly trust. SB3 uses pickle metadata.
    env = FleetEnv(scenario, meta["max_episode_steps"])
    # Rebuild Gym spaces locally for compatibility across NumPy versions.
    model = PPO.load(path, device="cpu", custom_objects={
        "observation_space": env.observation_space, "action_space": env.action_space,
        "_last_obs": None, "_last_episode_starts": None, "_last_original_obs": None,
    })
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise PlanningError("PPO checkpoint has incompatible observation/action dimensions")
    return model, env


def run_episode(model, env, *, seed=42):
    obs, _ = env.reset(seed=seed)
    ids = [v.id for v in env.sim.scenario.vehicles]
    paths = {key: [pos] for key, pos in zip(ids, env.sim.positions)}
    reward_sum = 0.0
    conflict_proposals = 0
    blocked_moves = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        conflict_proposals += info["conflict_proposals"]
        blocked_moves += info["blocked_moves"]
        for key, pos in zip(ids, env.sim.positions):
            paths[key].append(pos)
        if terminated or truncated:
            break
    for key, path in paths.items():
        while len(path) > 1 and path[-1] == path[-2]:
            path.pop()
    report = {"success": info["success"], "steps": info["steps"],
              "arrived": info["arrived"], "reward": round(reward_sum, 4),
              "conflict_proposals": conflict_proposals, "blocked_moves": blocked_moves}
    return paths, report


def plan_ppo(scenario, checkpoint):
    model, env = load_model(scenario, checkpoint)
    paths, report = run_episode(model, env)
    env.close()
    if not report["success"]:
        raise PlanningError(f"PPO timed out: {report['arrived']}/{len(scenario.vehicles)} vehicles arrived; train longer or use CBS")
    result = Result({k: tuple(p) for k, p in paths.items()}, "PPO with collision blocking", 0, scenario)
    result.validate()
    return result


def train_main():
    p = argparse.ArgumentParser(description="Train a scenario-specific PPO fleet policy on CPU")
    p.add_argument("scenario")
    p.add_argument("--output", default="outputs/ppo.zip")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=120)
    a = p.parse_args()
    path = train(load_scenario(a.scenario), a.output, steps=a.steps, seed=a.seed, max_steps=a.max_steps)
    print(f"Saved {path}. Evaluate before using this policy.")


def evaluate_main():
    p = argparse.ArgumentParser(description="Evaluate a PPO policy on its recorded scenario")
    p.add_argument("scenario")
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="outputs/evaluation.json")
    a = p.parse_args()
    scenario = load_scenario(a.scenario)
    model, env = load_model(scenario, a.model)
    paths, report = run_episode(model, env)
    if report["success"]:
        Result({k: tuple(v) for k, v in paths.items()}, "PPO", 0, scenario).validate()
    env.close()
    report["scope"] = "One deterministic rollout on the fixed training scenario; not a generalization benchmark"
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 2
