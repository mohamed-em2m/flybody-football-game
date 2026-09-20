# Scripts Directory

This folder contains training and rollout scripts for FlyBody RL experiments.

## Files

- **`football_policy.py`**: Shared linear policy utility module (`flatten_obs`, `linear_action`, `save_policy`, `load_policy`, `make_policy`).
- **`train_football_ars.py`**: Augmented Random Search (ARS) distributed trainer for `football_vs`.
- **`rollout_football_video.py`**: High-resolution video renderer for policy rollouts.
- **`watch_football_3d.py`**: Live interactive 3D MuJoCo viewer for watching matches.

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

### Watch a Match Live in 3D
Opens an interactive MuJoCo window (orbit/zoom, SPACE = pause, R = reset).
Score, possession and last event print to the console every 2s.
```bash
python -m scripts.watch_football_3d --opponent static --n-per-team 1
python -m scripts.watch_football_3d --opponent self_play --n-per-team 3 --formation-mode random --policy-npz runs/ars1/best.npz
```
Needs a display with OpenGL; without one it prints the mp4 fallback command.
