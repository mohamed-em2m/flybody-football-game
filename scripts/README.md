# Scripts Directory

This folder contains training and rollout scripts for FlyBody RL experiments.

## Files

- **`football_policy.py`**: Shared linear policy utility module (`flatten_obs`, `linear_action`, `save_policy`, `load_policy`).
- **`train_football_ars.py`**: Augmented Random Search (ARS) distributed trainer for `football_vs`.
- **`rollout_football_video.py`**: High-resolution video renderer for policy rollouts.

## Quickstart

### Train an ARS Football Agent
```bash
python scripts/train_football_ars.py --iters 5 --run-dir runs/ars1
```

### Resume Training
```bash
python scripts/train_football_ars.py --iters 5 --run-dir runs/ars1 --resume
```

### Render Policy Rollout to Video
```bash
python scripts/rollout_football_video.py --policy-npz runs/ars1/best.npz --out videos/football_rollout.mp4
```
