# RL_Mario

This is a personal reinforcement learning project to train an agent to play Super Mario. The environment is based on `gym_super_mario_bros`, and the algorithm is PPO from `Stable-Baselines3`. After baseline training and parameter tuning, the agent can reliably clear World 1-1 and World 1-2.

## Environment and Dependencies
- Python 3.8+
- See `requirements.txt` for all dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1) Train

```bash
python train_model.py
```

Training logs are written to `logs/` or `mario_logs/`. Model weights are saved as `ppo_mario.zip`, and the best-performing checkpoint is stored under `best_model/best_model.zip`.

2) Test/Evaluate

```bash
python test_model.py
```

For custom observation preprocessing, see `my_wrapper.py`. To quickly inspect observations only, use `test_obs.py`.

## Training Setup Overview
- Environment: `gym_super_mario_bros` (discrete action space), with NES rendering and standard preprocessing (grayscale, resize, frame skip, etc.)
- Wrappers: see `my_wrapper.py` (observation processing, action repeat, etc.)
- Algorithm: PPO (Stable-Baselines3)
- Logging: TensorBoard logs under `logs/` and `mario_logs/`

## Results
- After tuning and training, the agent can reliably clear:
  - World 1-1
  - World 1-2
- Best weights: `best_model/best_model.zip`; general checkpoint: `ppo_mario.zip`.

## Visualization and Reproduction Tips
- Launch TensorBoard to view training curves:

```bash
tensorboard --logdir logs | cat
```

- For full reproduction, set up the environment via `requirements.txt` and use the same seed, training steps, and hyperparameters. You can adjust hyperparameters and eval frequency directly in `train_model.py`.

## Acknowledgements
- Environment: `gym_super_mario_bros`
- RL framework: `Stable-Baselines3`
