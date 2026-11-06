"""ILOS command-line demo and machine-readable route export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from .planner import PlanningError, plan
from .scenario import load_scenario
from .visualize import save_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ilos", description="Collision-safe AGV fleet planning")
    parser.add_argument("scenario", help="Path to a JSON grid scenario")
    parser.add_argument("--output", default="outputs/plan.json", help="Route JSON output")
    parser.add_argument("--html", default="outputs/demo.html", help="Interactive self-contained HTML visualizer (empty to disable)")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=12000)
    parser.add_argument("--congestion-factor", type=float, default=1)
    parser.add_argument("--backend", choices=["cbs", "ppo"], default="cbs")
    parser.add_argument("--model", help="Trusted local PPO checkpoint for this scenario")
    args = parser.parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
        if args.backend == "ppo":
            if not args.model:
                raise PlanningError("--model is required for the PPO backend")
            try:
                from .learning import plan_ppo
            except ImportError as exc:
                raise PlanningError("Install the learning extra: pip install -e './modern[learning]'") from exc
            result = plan_ppo(scenario, args.model)
        else:
            result = plan(scenario, horizon=args.horizon, max_nodes=args.max_nodes,
                          congestion_factor=args.congestion_factor)
    except (PlanningError, OSError) as exc:
        print(f"Planning failed: {exc}", file=sys.stderr)
        return 2
    data = {
        "scenario": scenario.name, "algorithm": result.algorithm,
        "makespan": result.makespan, "total_steps": result.total_steps,
        "weighted_cost": result.weighted_cost, "expanded_nodes": result.expanded_nodes,
        "collision_free": True,
        "paths": {key: [list(p) for p in value] for key, value in result.paths.items()},
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.html:
        save_html(result, args.html)
    print(f"{len(result.paths)} AGVs routed without collisions | makespan={result.makespan} | total steps={result.total_steps}")
    print(f"JSON: {target}")
    if args.html:
        print(f"Visualizer: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
