# src/module2_rl/train.py

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from src.simulator.scenario_loader import load_scenario
from src.simulator.environment import MultiTruckGridEnv
from .rl_agent import DQNConfig, DQNAgent


def make_env(scenario_path: str) -> MultiTruckGridEnv:
    scenario = load_scenario(scenario_path)
    env = MultiTruckGridEnv(scenario=scenario, render_mode=None)
    return env


def train(
        scenario_path: str,
        total_training_steps: int = 200_000,
        max_episode_steps: Optional[int] = None,
        model_dir: str = "experiments/mlruns/dqn_multi_truck",
        save_interval: int = 10_000,
        device: str = "cuda",
):
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(scenario_path)
    obs, info = env.reset(seed=42)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    cfg = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_capacity=100_000,
        min_replay_size=2_000,
        target_update_interval=1_000,
        device=device,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100_000,
    )

    agent = DQNAgent(cfg)

    print(f"Starting training on device={agent.device} ...")
    print(f"state_dim={state_dim}, action_dim={action_dim}")

    episode = 0
    global_step = 0

    episode_reward = 0.0
    episode_length = 0

    start_time = time.time()

    while global_step < total_training_steps:
        state = obs
        action = agent.select_action(state, evaluate=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_obs, done)

        # Optimization
        loss = agent.optimize_model(global_step)

        episode_reward += reward
        episode_length += 1
        global_step += 1
        obs = next_obs

        if done or (max_episode_steps is not None and episode_length >= max_episode_steps):
            # Log
            elapsed = time.time() - start_time
            print(
                f"[Episode {episode}] "
                f"step={global_step} "
                f"reward={episode_reward:.2f} "
                f"length={episode_length} "
                f"epsilon={agent._get_epsilon():.3f} "
                f"time={elapsed:.1f}s"
            )

            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode += 1

        # Save model
        if global_step % save_interval == 0:
            model_path = os.path.join(model_dir, f"dqn_step_{global_step}.pt")
            agent.save(model_path)
            print(f"Saved checkpoint to {model_path}")

    # Final save
    final_path = os.path.join(model_dir, "dqn_final.pt")
    agent.save(final_path)
    print(f"Training finished. Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DQN on MultiTruckGridEnv.")
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="data/scenarios/example_scenario1.yaml",
        help="Path to scenario YAML.",
    )
    parser.add_argument(
        "--steps",
        "-t",
        type=int,
        default=200_000,
        help="Total training steps.",
    )
    parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        default="experiments/mlruns/dqn_multi_truck",
        help="Directory to save models.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    args = parser.parse_args()

    train(
        scenario_path=args.scenario,
        total_training_steps=args.steps,
        model_dir=args.model_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
