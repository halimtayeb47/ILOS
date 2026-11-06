# src/module2_rl/rl_agent.py

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_checkpoint(path, device="cpu"):
    """Read the recovered scalar metadata with a narrow NumPy allowlist."""
    scalar = np.core.multiarray.scalar
    allowed = [np.dtype, type(np.dtype("int64")), type(np.dtype("float64")),
               (scalar, "numpy._core.multiarray.scalar"),
               (scalar, "numpy.core.multiarray.scalar")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=device, weights_only=True)


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    min_replay_size: int = 1_000
    target_update_interval: int = 1_000
    device: str = "cuda"  # "cuda" or "cpu"
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
            self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )  # type: ignore
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        )

        self.policy_net = DQNNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        self.target_net = DQNNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.replay_capacity)

        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.min_replay_size = cfg.min_replay_size
        self.target_update_interval = cfg.target_update_interval

        # Epsilon-greedy
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay_steps = cfg.epsilon_decay_steps
        self.total_steps: int = 0

    # ----------------- Epsilon-greedy policy -----------------

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        self.total_steps += 1

        epsilon = self._get_epsilon() if not evaluate else 0.0

        if random.random() < epsilon:
            return random.randrange(self.cfg.action_dim)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action

    def _get_epsilon(self) -> float:
        # Linear decay
        fraction = min(float(self.total_steps) / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # ----------------- Replay buffer + optimization -----------------

    def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def can_optimize(self) -> bool:
        return len(self.replay_buffer) >= self.min_replay_size

    def optimize_model(self, global_step: int) -> Optional[float]:
        if not self.can_optimize():
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).float().to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states_t)
        state_action_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_values = rewards_t + self.gamma * (1.0 - dones_t) * max_next_q_values

        loss = F.mse_loss(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        if global_step % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ----------------- Save / Load -----------------

    def save(self, path: str) -> None:
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg.__dict__,
            "total_steps": self.total_steps,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, device: str = "cuda") -> "DQNAgent":
        # Load tensors and primitive metadata without arbitrary pickle objects.
        checkpoint = load_checkpoint(path, device=device)

        cfg_dict = checkpoint["cfg"]
        cfg = DQNConfig(**cfg_dict)
        cfg.device = device

        agent = DQNAgent(cfg)
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.total_steps = checkpoint.get("total_steps", 0)
        return agent
