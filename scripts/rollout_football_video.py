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
    from .football_policy import make_policy
except ImportError:  # `python scripts/rollout_football_video.py`
    from .football_policy import make_policy

import numpy as np  # noqa: E402  (after sys.path setup below)
import imageio.v2 as imageio  # noqa: E402


def draw_scoreboard(frame, task):
    """PIL scoreboard overlay: score, possession, last event, team names.

    MuJoCo has no text geoms, so jersey names/numbers live here (and in
    the GUI), keyed by fly name. Never breaks the render: any failure
    returns the frame untouched.
    """
    try:
        from PIL import Image, ImageDraw

        score = task.score
        poss = task.possession_side
        ev_name, ev_side, _ = task.last_event_info
        west_names = " ".join(w.name for w in task.west)
        east_names = " ".join(w.name for w in task.east)
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.rectangle([8, 8, 360, 78], fill=(0, 0, 0))
        d.text((14, 12), f"W {score['west']} : {score['east']} E", fill=(255, 255, 255))
        d.text(
            (14, 28),
            f"ball: {poss or '-'}   last: {ev_name}"
            + (f" ({ev_side})" if ev_side else ""),
            fill=(200, 200, 200),
        )
        d.text((14, 44), f"W: {west_names}", fill=(255, 110, 110))
        d.text((14, 60), f"E: {east_names}", fill=(110, 160, 255))
        return np.asarray(img)
    except Exception:
        return frame


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
    parser.add_argument("--n-per-team", type=int, default=1, help="Flies per side.")
    args = parser.parse_args()
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from flybody.fly_envs import football_vs

    env = football_vs(
        opponent_mode=args.opponent,
        time_limit=args.time_limit,
        n_per_team=args.n_per_team,
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
                draw_scoreboard(
                    env.physics.render(
                        height=args.height, width=args.width, camera_id=camera_id
                    ),
                    env.task,
                )
            )
        step += 1

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(
        f"saved {args.out}: {len(frames)} frames, "
        f"{step} steps, total_reward={total_reward:.2f}, "
        f"scored={env.task._scored}, score={env.task.score}"
    )


if __name__ == "__main__":
    main()
