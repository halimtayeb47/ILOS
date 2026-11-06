# src/integration/edge_server.py

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.simulator.scenario_loader import load_scenario
from src.simulator.environment import MultiTruckGridEnv
from src.module2_rl.rl_agent import DQNAgent

# --------- Config ---------

DEFAULT_MODEL_PATH = os.getenv(
    "ILOS_MODEL_PATH",
    "../checkpoints/dqn_final.pt",
)
DEFAULT_SCENARIO_PATH = os.getenv(
    "ILOS_SCENARIO_PATH",
    "data/scenarios/example_scenario1.yaml",
)
DEFAULT_DEVICE = os.getenv("ILOS_DEVICE", "cpu")


# --------- Request/Response Schemas ---------

class ActionRequest(BaseModel):
    # Optional raw state vector (direct from gym env)
    state: List[float] | None = None

    # أو تمثيل عالي المستوى (positions)، نوسّعه لاحقًا عند الدمج مع الرؤية
    truck_positions: Dict[str, Dict[str, int]] | None = None  # {"T0": {"x":0,"y":0}, ...}
    truck_goals: Dict[str, Dict[str, int]] | None = None
    obstacles: List[Dict[str, int]] | None = None

    # لو ما أرسل state، وفعّلنا وضع "rebuild_state_from_positions" يمكننا إعادة بناء observation
    rebuild_state: bool = False


class ActionResponse(BaseModel):
    action: int
    per_truck_actions: Dict[str, int]
    epsilon: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_path: str
    scenario_path: str


# --------- FastAPI App ---------

app = FastAPI(
    title="ILOS Edge RL Server",
    version="1.0.0",
    description="FastAPI edge server serving DQN actions for multi-truck logistics env.",
)


# --------- Lazy loaders ---------

@lru_cache(maxsize=1)
def get_env() -> MultiTruckGridEnv:
    scenario = load_scenario(DEFAULT_SCENARIO_PATH)
    # render_mode=None لأننا ما نحتاج طباعة على السيرفر
    env = MultiTruckGridEnv(scenario=scenario, render_mode=None)
    return env


@lru_cache(maxsize=1)
def get_agent() -> DQNAgent:
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise RuntimeError(f"Model file not found at {DEFAULT_MODEL_PATH}")

    device = DEFAULT_DEVICE
    agent = DQNAgent.load(DEFAULT_MODEL_PATH, device=device)
    return agent


# --------- Helpers ---------

def decode_joint_action(env: MultiTruckGridEnv, action: int) -> Dict[str, int]:
    """
    نفس المنطق داخل env._decode_action لكن متاح هنا عشان نرجع per_truck_actions.
    0=stay,1=up,2=down,3=left,4=right
    """
    a = int(action)
    base = env.n_actions_per_truck
    per_truck = {}
    for truck_id in env.truck_ids:
        per_truck[truck_id] = a % base
        a //= base
    return per_truck


def build_state_from_positions(req: ActionRequest, env: MultiTruckGridEnv) -> np.ndarray:
    """
    إعادة بناء observation من truck_positions / truck_goals / obstacles
    لو أردنا دمج مع رؤية خارجية لاحقًا.
    الآن سنفترض أن truck_ids تطابق T0, T1, ... الموجودة في السيناريو.
    """
    w = env.world.width
    h = env.world.height

    trucks_features: List[float] = []
    # positions
    if req.truck_positions is None or req.truck_goals is None:
        raise HTTPException(
            status_code=400,
            detail="truck_positions and truck_goals must be provided when rebuild_state=True",
        )

    for truck_id in env.truck_ids:
        if truck_id not in req.truck_positions or truck_id not in req.truck_goals:
            raise HTTPException(
                status_code=400,
                detail=f"Missing position/goal for truck_id={truck_id}",
            )

        p = req.truck_positions[truck_id]
        g = req.truck_goals[truck_id]
        x = p["x"]
        y = p["y"]
        gx = g["x"]
        gy = g["y"]

        x_norm = x / max(w - 1, 1)
        y_norm = y / max(h - 1, 1)
        gx_norm = gx / max(w - 1, 1)
        gy_norm = gy / max(h - 1, 1)
        done_flag = 0.0  # حالياً نفترض الكل غير منتهٍ

        trucks_features.extend([x_norm, y_norm, gx_norm, gy_norm, done_flag])

    obstacle_grid = np.zeros((h, w), dtype=np.float32)
    if req.obstacles is not None:
        for o in req.obstacles:
            ox = o["x"]
            oy = o["y"]
            if 0 <= ox < w and 0 <= oy < h:
                obstacle_grid[oy, ox] = 1.0

    obs = np.concatenate(
        [np.array(trucks_features, dtype=np.float32), obstacle_grid.flatten()]
    )
    return obs


# --------- Routes ---------

@app.get("/health", response_model=HealthResponse)
def health_check():
    # مجرد فحص جاهزية
    device = DEFAULT_DEVICE
    return HealthResponse(
        status="ok",
        device=device,
        model_path=DEFAULT_MODEL_PATH,
        scenario_path=DEFAULT_SCENARIO_PATH,
    )


@app.post("/act", response_model=ActionResponse)
def act(req: ActionRequest):
    env = get_env()
    agent = get_agent()

    # تحديد state
    if req.state is not None:
        state_np = np.array(req.state, dtype=np.float32)
        expected_dim = env.observation_space.shape[0]
        if state_np.shape[0] != expected_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state dimension {state_np.shape[0]}, expected {expected_dim}",
            )
    elif req.rebuild_state:
        state_np = build_state_from_positions(req, env)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'state' must be provided or 'rebuild_state=True' with positions.",
        )

    action = agent.select_action(state_np, evaluate=True)
    per_truck_actions = decode_joint_action(env, action)

    return ActionResponse(
        action=action,
        per_truck_actions=per_truck_actions,
        epsilon=0.0,  # evaluate mode
    )
