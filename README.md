# ILOS - Intelligent Logistics Optimization System

**Huawei ICT Competition 2025-2026**  
**Islamic University of Madinah | FCIS Team**

## Overview

ILOS (Intelligent Logistics Optimization System) is a centralized multi-agent path planning solution for automated guided vehicles (AGVs) in warehouses and ports. The system uses Deep Q-Learning with MindSpore to coordinate multiple vehicles simultaneously, ensuring collision-free and optimal routing.

## Key Features

- Centralized Deep Q-Network (DQN) agent for multi-vehicle coordination
- Built with MindSpore 2.x framework
- Custom gymnasium-based simulation environment
- Collision-free path planning with 100% success rate
- Modular architecture designed for future expansion

## Project Structure

```
ILOS_Complete/
├── src/
│   ├── simulator/           # Grid environment and entities
│   │   ├── environment.py   # MultiTruckGridEnv (Gymnasium)
│   │   ├── entities.py      # Truck and Destination classes
│   │   └── scenario_loader.py
│   ├── module2_rl/          # Reinforcement Learning core
│   │   ├── rl_agent_mindspore.py  # DQN Agent implementation
│   │   ├── train_mindspore.py     # Training script
│   │   └── evaluate_mindspore.py  # Evaluation script
│   ├── module1_vision/      # Vision module (architecture defined)
│   ├── module3_ar/          # AR visualization module (architecture defined)
│   └── integration/         # Edge server for external communication
├── data/
│   └── scenarios/           # Training scenarios
│       └── simple_2trucks.yaml
├── experiments/             # Trained models and checkpoints
├── scripts/                 # Utility scripts
└── tests/                   # Unit tests
```

## System Requirements

- Python 3.8 or higher
- MindSpore 2.0 or higher
- CUDA-compatible GPU (recommended) or CPU

## Installation

### 1. Download the Project

Download and extract the ZIP file from GitHub:

```bash
unzip ILOS_Complete.zip
cd ILOS_Complete
```

### 2. Set Up Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

For development features:

```bash
pip install -e .
```

## Usage

### Training

Train the DQN agent on a scenario:

```bash
python -m src.module2_rl.train_mindspore \
    --scenario data/scenarios/simple_2trucks.yaml \
    --steps 200000 \
    --device GPU \
    --save_dir experiments/my_training
```

Training parameters:
- `--scenario`: Path to scenario YAML file
- `--steps`: Total training steps (default: 200,000)
- `--device`: CPU or GPU (default: GPU) (CPU is recommeneded for it's broader compatabillity as GPU is only on Linux)
- `--save_dir`: Directory to save model checkpoints
- `--save_interval`: Save checkpoint every N steps (default: 20,000)

### Evaluation

Evaluate a trained model:

```bash
python -m src.module2_rl.evaluate_mindspore \
    --model experiments/my_training/dqn_final.ckpt \
    --scenario data/scenarios/simple_2trucks.yaml \
    --episodes 20 \
    --device GPU
```

Evaluation parameters:
- `--model`: Path to trained model checkpoint
- `--scenario`: Path to scenario YAML file
- `--episodes`: Number of test episodes (default: 20)
- `--device`: CPU or GPU (default: GPU)

### Quick Evaluation

Use the quick evaluation script for fast testing:

```bash
python scripts/quick_eval.py \
    --model experiments/my_training/dqn_final.ckpt \
    --scenario data/scenarios/simple_2trucks.yaml
```

## Using the Simple 2-Truck Scenario

The project includes a validated scenario in `data/scenarios/simple_2trucks.yaml`:
- 8x8 grid environment
- 2 trucks with opposing diagonal paths
- 4 central obstacles creating a challenging routing problem
- Maximum 60 steps per episode

This scenario is what the trained models were validated on, achieving 100% success rate. Use this scenario for training and evaluation to ensure reliable results.

## Algorithm Details

### Deep Q-Network (DQN)

- **State Space**: 46-dimensional vector encoding positions, goals, grid occupancy, and distances
- **Action Space**: Joint action space controlling all trucks simultaneously (5 actions per truck: N/S/E/W/Stay)
- **Network Architecture**: Fully connected layers with 256 hidden units
- **Optimization**: Huber loss with ultra-low learning rate (5e-5) for stability

### Key Innovations

1. **Centralized Coordination**: Single agent controls all vehicles to prevent conflicts
2. **Stable Training**: Ultra-low learning rate and Huber loss for convergence
3. **Safety-First Rewards**: Massive penalties for collisions ensure safe behavior

## Performance Metrics

On the 2-truck scenario:
- Success Rate: 100% (20/20 episodes)
- Optimal Steps: 10 (mathematical minimum)
- Average Reward: +153.00
- Collision Rate: 0%
- Standard Deviation: 0.00

## Future Development

The system architecture includes placeholders for:
- **Module 1**: Computer vision pipeline for real-world vehicle detection
- **Module 3**: AR visualization interface for human operators
- **Edge Deployment**: FastAPI server for hardware integration

## License

This project was developed for the Huawei ICT Competition 2025-2026.

## Contact

Islamic University of Madinah | FCIS Team