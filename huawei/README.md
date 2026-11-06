# Huawei backend and recovered source

This folder keeps the historical DQN work separate from the modern PPO and CBS
implementation.

## What was recovered

The uploaded archive contains **PyTorch DQN source and `.pt` checkpoints**, a
three-vehicle 8×8 scenario, the grid simulator, an edge API and simulated vision
input. Earlier repository documentation described a two-vehicle MindSpore model,
but its original MindSpore source and `.ckpt` weights were not present in the
archive. The two versions must not be treated as the same recovered model.

| Folder | Contents |
| --- | --- |
| `legacy/` | Recovered Python source with the maintenance changes below |
| `checkpoints/dqn_final.pt` | Original final PyTorch checkpoint, unchanged |
| `mindspore_backend/` | New MindSpore implementation of the recovered three-layer DQN |
| `source-manifest.json` | SHA-256 hashes of the recovered input files |
| `tests/` | Checkpoint, timeout, training and save/load tests |

The MindSpore implementation is a **compatibility port**, not recovered historical
code. It can train a model or convert the supplied PyTorch network's parameters
without inventing missing training results. The recovered DQN has 79 input
features, two 256-unit hidden layers and 125 joint-action outputs.

## Run the recovered PyTorch model

Use a separate virtual environment for this backend.

```bash
python -m pip install -r huawei/legacy/requirements.txt
cd huawei/legacy
python -m src.module2_rl.evaluate --model ../checkpoints/dqn_final.pt --device cpu --episodes 1
```

This checkpoint does **not** solve the supplied scenario in the measured rollout:
0/3 vehicles reached their goals within 60 steps. It is retained for reproducible
baseline research, not advertised as a validated routing policy. The old
simulator penalizes overlaps but does not prevent them or enforce head-on swaps.
Use the modern simulator for collision-blocked trajectories.

The retained `DummyVisionDetector` supplies simulated positions; it is not an
image-recognition model. No functioning AR implementation was present, so empty
AR files were excluded.

## MindSpore setup

Install the framework in an environment supported by the
[official installation guide](https://www.mindspore.cn/install/en), then:

```bash
python -m pip install -r huawei/requirements.txt
cd huawei
python -m mindspore_backend train --steps 100000 --model outputs/dqn.ckpt
python -m mindspore_backend evaluate --model outputs/dqn.ckpt
```

To convert the supplied parameters, also install PyTorch:

```bash
python -m pip install 'torch>=2.6,<3'
python -m mindspore_backend convert --source checkpoints/dqn_final.pt --model outputs/converted.ckpt
python -m mindspore_backend evaluate --model outputs/converted.ckpt
```

A converted model preserves the old weights; conversion does not improve the
policy. Every `.ckpt` has a matching JSON file recording network dimensions.
Training uses replay, a target network, Huber loss and gradient clipping. Save
files contain inference weights, not a resumable optimizer/replay snapshot.

**Validation:** the recovered PyTorch model and timeout tests run locally. The
MindSpore native runtime exits with a bus error during import in the development
container, before the project code executes. The port therefore requires the
separate MindSpore CI job to establish runtime validation on a supported host.

## Maintenance changes

- Restrict PyTorch loading to tensors, primitive metadata and explicitly allowed
  NumPy scalar types; no unrestricted pickle fallback.
- Timeouts no longer mark undelivered trucks as completed.
- Trucks starting at their goal are marked arrived on reset.
- Reject out-of-range joint actions and report actual goal completion.
- Update the edge API's default checkpoint path for the new folder structure.
- Keep only runtime dependencies used by the recovered code.
- Normalize source line endings and restore package initializer files.

Local environments, editor settings, private application forms, screenshots of
the development desktop, empty placeholders and redundant intermediate
checkpoints are excluded. The source manifest refers to input hashes, so files
with the listed maintenance changes intentionally have different current hashes.
