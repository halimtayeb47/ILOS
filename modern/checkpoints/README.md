# Three-vehicle PPO model

`port_ppo.zip` is a trained PPO policy for `../examples/three_vehicles.json`.
`port_ppo.json` records its seed, training steps, map fingerprint and checksum.

| Property | Value |
| --- | --- |
| Policy | PPO, separate 128–128 actor and value networks |
| Training | 100,096 environment steps; requested 100,000, seed 42 |
| Actions | Five choices per vehicle; 15 policy logits for three vehicles |
| Observation | Positions, goals, arrival flags, obstacles and congestion |
| Fixed-map result | 3/3 arrived in 14 steps |
| Conflict proposals / blocked moves | 0 / 0 |
| Validation | Independent full-route collision and adjacency checks |

This is a small demonstration policy, trained and evaluated on the **same fixed
map**. It is not a general warehouse model, a pretrained vision model or a
physical controller. Different starts, goals, obstacles, congestion or fleet size
require training a new checkpoint. No held-out generalization result is claimed.

The recovered DQN reached 0/3 goals within 60 steps on the supplied map. The same
DQN also reached 0/3 when evaluated with the modern collision blocking. These are
observations of the supplied checkpoints, not a controlled comparison of all
DQN and PPO configurations: training budgets and simulator dynamics differ.

The PPO model was trained using Python 3.12, PyTorch 2.15.0, Stable-Baselines3
2.9.2 and Gymnasium 1.4.0 on CPU. The loader rebuilds Gym spaces to support both
NumPy 1.x and 2.x. Load only trusted checkpoints; metadata checksums alone do not
establish trust.
