"""MindSpore DQN with the recovered model's layer names and dimensions."""
from __future__ import annotations
import json
from pathlib import Path
import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np

class Network(nn.Cell):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Dense(state_dim, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, hidden_dim)
        self.fc3 = nn.Dense(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=5e-5, gamma=0.99):
        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, lr=lr, gamma=gamma)
        self.policy = Network(state_dim, action_dim, hidden_dim)
        self.target = Network(state_dim, action_dim, hidden_dim)
        self.sync_target()
        self.target.set_train(False)
        self.optimizer = nn.Adam(self.policy.trainable_params(), learning_rate=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.grad_fn = ms.value_and_grad(self.loss, None, self.optimizer.parameters)

    def sync_target(self):
        for dest, source in zip(self.target.get_parameters(), self.policy.get_parameters()):
            dest.set_data(source.data)

    def loss(self, states, actions, targets):
        chosen = ops.gather_d(self.policy(states), 1, actions.reshape((-1, 1))).reshape((-1,))
        return self.loss_fn(chosen, targets)

    def act(self, state):
        values = self.policy(Tensor(np.asarray(state, dtype=np.float32)[None]))
        return int(values.asnumpy().argmax(axis=1)[0])

    def update(self, batch):
        states, actions, rewards, next_states, dones = [np.asarray(x) for x in zip(*batch)]
        next_q = self.target(Tensor(next_states.astype(np.float32))).asnumpy().max(axis=1)
        targets = rewards + self.config['gamma'] * (1 - dones.astype(np.float32)) * next_q
        loss, grads = self.grad_fn(Tensor(states.astype(np.float32)), Tensor(actions.astype(np.int32)),
                                  Tensor(targets.astype(np.float32)))
        self.optimizer(ops.clip_by_global_norm(grads, 10.0))
        return float(loss.asnumpy())

    def save(self, path):
        path = Path(path).with_suffix('.ckpt')
        path.parent.mkdir(parents=True, exist_ok=True)
        ms.save_checkpoint(self.policy, str(path))
        path.with_suffix('.json').write_text(json.dumps(self.config, indent=2) + '\n', encoding='utf-8')
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        agent = cls(**json.loads(path.with_suffix('.json').read_text(encoding='utf-8')))
        missing, unused = ms.load_param_into_net(agent.policy, ms.load_checkpoint(str(path)), strict_load=True)
        if missing or unused:
            raise ValueError('Checkpoint parameters do not match this network')
        agent.sync_target()
        return agent
