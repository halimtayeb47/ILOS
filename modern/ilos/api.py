"""Local planning API. Model paths are configured by the operator, never clients."""
from __future__ import annotations
import hmac
import os
from typing import Literal
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from .planner import PlanningError, plan
from .scenario import from_dict

app = FastAPI(title='ILOS Planning API', version='0.2.0')

class VehicleInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=80)
    start: tuple[StrictInt, StrictInt]
    goal: tuple[StrictInt, StrictInt]

class CongestionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cell: tuple[StrictInt, StrictInt]
    cost: float = Field(ge=0, le=1000, allow_inf_nan=False)

class ScenarioInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default='ILOS scenario', max_length=200)
    width: StrictInt = Field(ge=1, le=32)
    height: StrictInt = Field(ge=1, le=32)
    vehicles: list[VehicleInput] = Field(min_length=1, max_length=8)
    obstacles: list[tuple[StrictInt, StrictInt]] = Field(default_factory=list, max_length=1024)
    congestion: list[CongestionInput] = Field(default_factory=list, max_length=1024)

class PlanInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scenario: ScenarioInput
    backend: Literal['cbs', 'ppo'] = 'cbs'
    horizon: StrictInt = Field(default=128, ge=0, le=256)
    max_nodes: StrictInt = Field(default=1000, ge=1, le=5000)

def authorize(x_api_key: str | None = Header(default=None)):
    expected = os.getenv('ILOS_API_KEY')
    if expected and (not x_api_key or not hmac.compare_digest(expected.encode(), x_api_key.encode())):
        raise HTTPException(401, 'A valid X-API-Key is required')

@app.get('/health')
def health():
    return {'status': 'ok', 'version': '0.2.0', 'backends': ['cbs', 'ppo'],
            'ppo_configured': bool(os.getenv('ILOS_PPO_MODEL')), 'mode': 'simulation'}

@app.post('/plan', dependencies=[Depends(authorize)])
def make_plan(request: PlanInput):
    try:
        scenario = from_dict(request.scenario.model_dump())
        if request.backend == 'ppo':
            path = os.getenv('ILOS_PPO_MODEL')
            if not path:
                raise HTTPException(503, 'Set ILOS_PPO_MODEL to a trained local checkpoint')
            try:
                from .learning import plan_ppo
            except ImportError as exc:
                raise HTTPException(503, 'Install the learning extra to use PPO') from exc
            result = plan_ppo(scenario, path)
        else:
            result = plan(scenario, horizon=request.horizon, max_nodes=request.max_nodes, timeout=5)
        result.validate()
        return {'algorithm': result.algorithm, 'collision_free': True,
                'makespan': result.makespan, 'total_steps': result.total_steps,
                'weighted_cost': result.weighted_cost, 'paths': result.paths}
    except PlanningError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, 'Configured model could not be loaded; check the server configuration') from exc
