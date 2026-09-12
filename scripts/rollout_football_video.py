"""Roll out football_vs and save a video of the episode.

Usage:
    python rollout_football_video.py [--out football_rollout.mp4]
                                     [--time-limit 4.0] [--seed 0]
                                     [--opponent static] [--fps 30]

After training, point --policy-npz to a saved policy checkpoint (e.g.
runs/ars1/best.npz); the rendering loop will load and record the
trained behavior.
"""

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # `python -m scripts.rollout_football_video`
    from .football_policy import flatten_obs, linear_action, load_policy
except ImportError:  # `python scripts/rollout_football_video.py`
    from football_policy import flatten_obs, linear_action, load_policy

import numpy as np
import imageio.v2 as imageio


def make_policy(n_act, seed=0, policy_npz=None):
    """Random actions, or a trained linear policy from train_football_ars."""
    if policy_npz is None:
        rng = np.random.RandomState(seed)

        def policy(observation):
            del observation  # Unused by random policy.
            return rng.uniform(-0.5, 0.5, n_act)

        return policy

    matrix, mean, var, keys, minimum, maximum = load_policy(policy_npz)

    def policy(observation):
        flat = flatten_obs(observation, keys)
        return linear_action(flat, matrix, mean, var, minimum, maximum)

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="videos/football_rollout.mp4")
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opponent", default="static", choices=["static", "scripted", "self_play"]
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--render-every", type=int, default=4, help="Render every N-th control step."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--policy-npz",
        default=None,
        help="Trained policy from train_football_ars.py (runs/<name>/best.npz).",
    )
    args = parser.parse_args()
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from flybody.fly_envs import football_vs

    env = football_vs(
        opponent_mode=args.opponent,
        time_limit=args.time_limit,
        random_state=np.random.RandomState(args.seed),
    )
    n_act = env.action_spec().shape[0]
    policy = make_policy(n_act, seed=args.seed + 1, policy_npz=args.policy_npz)

    timestep = env.reset()
    # Ball-tracking overview camera added by FootballArena; fall back to
    # the default camera if it is missing.
    camera_id = 0
    for cam_name in ("overview", "football/overview"):
        try:
            camera_id = env.physics.model.name2id(cam_name, "camera")
            break
        except KeyError:
            pass
    frames = []
    total_reward = 0.0
    step = 0
    while not timestep.last():
        action = policy(timestep.observation)
        timestep = env.step(action)
        total_reward += float(timestep.reward)
        if step % 1000 == 0:
            print(f"step {step}, t={env.physics.time():.2f}s", flush=True)
        if step % args.render_every == 0:
            frames.append(
                env.physics.render(
                    height=args.height, width=args.width, camera_id=camera_id
                )
            )
        step += 1

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(
        f"saved {args.out}: {len(frames)} frames, "
        f"{step} steps, total_reward={total_reward:.2f}, "
        f"scored={env.task._scored}"
    )


if __name__ == "__main__":
    main()
