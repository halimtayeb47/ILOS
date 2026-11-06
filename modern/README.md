# Modern ILOS

Two complementary backends share the same JSON maps and route validator:

- **PPO:** a learned actor–critic policy built with PyTorch and Stable-Baselines3.
  Each vehicle has five actions. The policy produces `5 × N` action logits instead
  of the legacy DQN's `5^N` joint-action Q values. This is an architectural change,
  not a claim of universally higher route quality.
- **CBS:** Conflict-Based Search with space-time A*. It needs no training and is
  the default for immediate, reproducible route planning.

## Run the six-vehicle demo

Run all commands from the repository root with Python 3.10+.

```bash
python -m pip install -e ./modern
ilos modern/examples/six_vehicles.json --html outputs/demo.html
```

Open `outputs/demo.html`. It works offline and includes play, pause, restart and a
time slider. JSON routes are saved to `outputs/plan.json`.

## Run the trained PPO example

```bash
python -m pip install -e './modern[learning]'
ilos modern/examples/three_vehicles.json --backend ppo --model modern/checkpoints/port_ppo.zip --html outputs/ppo.html
```

See the [model card](checkpoints/README.md) for its measured result and scope.
The checkpoint is specific to the included three-vehicle map. A map, goal,
congestion or fleet change requires retraining; incompatible metadata is rejected.

## Train and evaluate your own model

```bash
ilos-train modern/examples/three_vehicles.json --steps 100000 --seed 42 --max-steps 60 --output outputs/ppo.zip
ilos-evaluate modern/examples/three_vehicles.json --model outputs/ppo.zip --output outputs/evaluation.json
ilos modern/examples/three_vehicles.json --backend ppo --model outputs/ppo.zip --html outputs/ppo.html
```

Training writes weights plus JSON metadata with the scenario fingerprint, seed,
step count and checkpoint checksum. Evaluation measures actual goal completion;
timeouts return a nonzero exit code. Train with multiple seeds and representative
maps before making performance claims. The supplied evaluation is one
deterministic rollout on the fixed training map, not a held-out benchmark.

The PPO simulator blocks same-cell conflicts and head-on swaps. Blocking is
repeated until stopped vehicles cannot cause another collision. The report
includes conflict proposals so this protection does not hide a poor policy.
Vehicles remain parked at their destinations. All exported routes pass the same
independent adjacency, obstacle, start/goal and collision checks as CBS.

## Local API

```bash
python -m pip install -e './modern[api]'
python -m uvicorn ilos.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. `GET /health` reports configuration;
`POST /plan` accepts `scenario`, `backend`, `horizon` and `max_nodes`.

```json
{
  "scenario": {
    "width": 4, "height": 3,
    "obstacles": [[1, 1]],
    "vehicles": [
      {"id": "A", "start": [0, 0], "goal": [3, 2]},
      {"id": "B", "start": [3, 0], "goal": [0, 2]}
    ]
  },
  "backend": "cbs"
}
```

For PPO, install the `learning` extra and set `ILOS_PPO_MODEL` in the server's
environment. To require authentication, set `ILOS_API_KEY`; clients then send
`X-API-Key`. `.env.example` documents the variables; the application does not
load `.env` files automatically. Returned paths start at time zero. Failed
planning returns HTTP 422, missing model configuration HTTP 503.

## Scenario format

Coordinates are integer `[x, y]` cells. Starts and goals must be unique, in bounds
and unblocked. Optional congestion entries have the form
`{"cell": [2, 1], "cost": 3.0}`. Every move or wait takes one discrete time step.
CBS minimizes summed travel costs plus nonnegative cell penalties within the
configured horizon, when its search finishes. It can grow exponentially on
crowded maps; exhausted budgets return errors, never unverified routes.

CLI limits: `--horizon 200`, `--max-nodes 20000`,
`--congestion-factor 2`, and `--html ''` to skip playback.

## Tests and references

```bash
python -m pip install -e './modern[test,api,learning]'
python -m pytest modern/tests -q
```

- [PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Conflict-Based Search paper](https://ojs.aaai.org/index.php/AAAI/article/view/8140)

The grid model does not represent uncertain sensing, vehicle sizes, braking,
continuous movement or a certified physical safety controller.
