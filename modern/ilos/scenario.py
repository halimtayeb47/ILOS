"""Strict JSON input and canonical scenario serialization."""
from __future__ import annotations

import json
import math
from pathlib import Path
from .planner import PlanningError, Scenario, Vehicle


def cell(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(type(v) is int for v in value):
        raise PlanningError(f"Expected [integer_x, integer_y], got {value!r}")
    return value[0], value[1]


def from_dict(obj):
    try:
        if not isinstance(obj, dict):
            raise PlanningError("Scenario must be an object")
        if type(obj['width']) is not int or type(obj['height']) is not int:
            raise PlanningError("Width and height must be integers")
        if not isinstance(obj['vehicles'], list):
            raise PlanningError("Vehicles must be a list")
        vehicles = []
        for v in obj['vehicles']:
            if not isinstance(v['id'], str) or not v['id'].strip() or len(v['id']) > 80:
                raise PlanningError("Vehicle IDs must be nonempty strings up to 80 characters")
            vehicles.append(Vehicle(v['id'], cell(v['start']), cell(v['goal'])))
        weights = {}
        for e in obj.get('congestion', []):
            w = e['cost']
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
                raise PlanningError("Congestion costs must be finite nonnegative numbers")
            weights[cell(e['cell'])] = float(w)
        return Scenario(obj['width'], obj['height'], tuple(vehicles),
                        frozenset(cell(p) for p in obj.get('obstacles', [])), weights,
                        str(obj.get('name', 'ILOS scenario'))[:200])
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PlanningError):
            raise
        raise PlanningError(f"Invalid scenario: {exc}") from exc


def load_scenario(path):
    try:
        return from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
    except json.JSONDecodeError as exc:
        raise PlanningError(f"Invalid scenario JSON: {exc}") from exc


def scenario_dict(scenario):
    return {'name': scenario.name, 'width': scenario.width, 'height': scenario.height,
            'obstacles': [list(c) for c in sorted(scenario.obstacles)],
            'congestion': [{'cell': list(c), 'cost': w} for c, w in sorted(scenario.congestion.items())],
            'vehicles': [{'id': v.id, 'start': list(v.start), 'goal': list(v.goal)} for v in scenario.vehicles]}
