# Scripts Directory

This folder contains training and rollout scripts for FlyBody RL experiments.

## Files

- **`football_policy.py`**: Shared linear policy utility module (`flatten_obs`, `linear_action`, `save_policy`, `load_policy`).
- **`train_football_ars.py`**: Augmented Random Search (ARS) distributed trainer for `football_vs`.
- **`rollout_football_video.py`**: High-resolution video renderer for policy rollouts.

## Quickstart
Run from the repo root as a package (`scripts/` uses relative imports):
```bash
python -m scripts.train_football_ars --iters 5 --run-dir runs/ars1
```

### Resume Training
```bash
python -m scripts.train_football_ars --iters 5 --run-dir runs/ars1 --resume
```

### Render Policy Rollout to Video
```bash
python -m scripts.rollout_football_video --policy-npz runs/ars1/best.npz --out videos/football_rollout.mp4
```
