"""Reproduce the supplied DQN rollouts; run after installing modern[learning] and PyYAML."""
import sys,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'huawei/legacy'))
import torch
from src.module2_rl.rl_agent import DQNAgent
from src.simulator.environment import MultiTruckGridEnv
from src.simulator.scenario_loader import load_scenario
from ilos.planner import first_conflict
from ilos.simulation import FleetSimulator
from ilos.scenario import load_scenario as load_modern

a=DQNAgent.load(ROOT / 'huawei/checkpoints/dqn_final.pt',device='cpu');torch.set_num_threads(1)
e=MultiTruckGridEnv(load_scenario(str(ROOT / 'huawei/legacy/data/scenarios/example_scenario1.yaml')))
o,_=e.reset();paths={i:[t.position] for i,t in e.trucks.items()}
for step in range(60):
 o,r,d,t,info=e.step(a.select_action(o,evaluate=True))
 for i,truck in e.trucks.items(): paths[i].append(truck.position)
 if d or t:break
report={'algorithm':'Recovered PyTorch DQN','success':info['success'],'arrived':sum(t.position==e.scenario.truck_goals[k] for k,t in e.trucks.items()),'steps':step+1,'first_conflict':first_conflict(paths),'state_dim':a.cfg.state_dim,'action_dim':int(a.cfg.action_dim),'paths':paths,'scope':'One deterministic rollout on the supplied three-vehicle scenario using the legacy simulator with timeout correction'}
Path(ROOT / 'docs/results/recovered_dqn.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='paths'})
s=load_modern(ROOT / 'modern/examples/three_vehicles.json');sim=FleetSimulator(s,max_steps=60);o=sim.reset();total_conflicts=0
for step in range(60):
 action=a.select_action(__import__('numpy').asarray(o[:a.cfg.state_dim],dtype='float32'),evaluate=True)
 moves=[]
 for _ in s.vehicles:moves.append(action%5);action//=5
 o,r,d,t,info=sim.step(moves);total_conflicts+=info['conflict_proposals']
 if d or t:break
report={'algorithm':'Recovered DQN on modern dynamics','success':info['success'],'arrived':info['arrived'],'steps':step+1,'conflict_proposals':total_conflicts,'scope':'Same map, goals and collision blocking used for PPO evaluation; recovered DQN was trained with the older dynamics'}
Path(ROOT / 'docs/results/recovered_dqn_modern_dynamics.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
