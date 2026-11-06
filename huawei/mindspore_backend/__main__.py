"""Train, evaluate or convert the recovered DQN. Run from huawei/."""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys
import numpy as np
import mindspore as ms
from .agent import Agent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'legacy'))
from src.simulator.environment import MultiTruckGridEnv
from src.simulator.scenario_loader import load_scenario


def main():
    p = argparse.ArgumentParser(description='ILOS MindSpore DQN backend')
    p.add_argument('command', choices=['train', 'evaluate', 'convert'])
    p.add_argument('--scenario', default=str(ROOT / 'legacy/data/scenarios/example_scenario1.yaml'))
    p.add_argument('--model', default=str(ROOT / 'checkpoints/dqn_final.ckpt'))
    p.add_argument('--source', default=str(ROOT / 'checkpoints/dqn_final.pt'))
    p.add_argument('--steps', type=int, default=100_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['CPU', 'GPU', 'Ascend'], default='CPU')
    a = p.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=a.device)
    ms.set_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    if a.command == 'convert':
        from src.module2_rl.rl_agent import load_checkpoint
        checkpoint = load_checkpoint(a.source)
        cfg = checkpoint['cfg']
        agent = Agent(cfg['state_dim'], cfg['action_dim'], cfg['hidden_dim'])
        params = {k: ms.Parameter(ms.Tensor(v.detach().cpu().numpy()), name=k)
                  for k, v in checkpoint['policy_state_dict'].items()}
        missing, unused = ms.load_param_into_net(agent.policy, params, strict_load=True)
        if missing or unused:
            raise ValueError(f'Incompatible checkpoint: missing={missing}, unused={unused}')
        print(agent.save(a.model))
        return
    env = MultiTruckGridEnv(load_scenario(a.scenario))
    obs, _ = env.reset(seed=a.seed)
    if a.command == 'train':
        if a.steps < 1:
            p.error('--steps must be positive')
        agent = Agent(env.observation_space.shape[0], env.action_space.n)
        memory = deque(maxlen=100_000)
        for step in range(a.steps):
            epsilon = max(0.05, 1.0 - step / max(a.steps * 0.8, 1))
            action = random.randrange(env.action_space.n) if random.random() < epsilon else agent.act(obs)
            nxt, reward, done, timeout, _ = env.step(action)
            memory.append((obs, action, reward, nxt, done or timeout))
            if len(memory) >= 256:
                agent.update(random.sample(list(memory), 64))
            if (step + 1) % 1000 == 0:
                agent.sync_target()
            obs = env.reset()[0] if done or timeout else nxt
        print(agent.save(a.model))
    else:
        agent = Agent.load(a.model)
        if agent.config['state_dim'] != len(obs) or agent.config['action_dim'] != env.action_space.n:
            raise ValueError('Checkpoint dimensions do not match the scenario')
        reward_sum = 0.0
        conflicts = 0
        for step in range(env.world.max_steps):
            obs, reward, done, timeout, info = env.step(agent.act(obs))
            reward_sum += reward
            conflicts += info['conflicts']
            if done or timeout:
                break
        success = all(t.position == env.scenario.truck_goals[k] for k, t in env.trucks.items())
        print(json.dumps({'success': success, 'steps': step + 1, 'reward': reward_sum,
                          'vertex_conflicts': conflicts, 'scope': 'Legacy simulator; no head-on swap enforcement'}, indent=2))
        if not success:
            raise SystemExit(2)
    env.close()

if __name__ == '__main__':
    main()
