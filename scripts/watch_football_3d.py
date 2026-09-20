"""Watch football_vs live in an interactive 3D MuJoCo viewer.

Usage:
    python -m scripts.watch_football_3d [--opponent self_play]
        [--n-per-team 3] [--formation-mode random]
        [--policy-npz runs/ars1/best.npz] [--time-limit 120] [--seed 0]

Viewer controls: left-drag = orbit, right-drag = pan, scroll = zoom,
SPACE = pause, H = help overlay, R = reset episode. Switch cameras
(e.g. the ball-tracking overview cam) from the camera menu.

After training, point --policy-npz to a saved policy checkpoint (e.g.
runs/ars1/best.npz); without it every fly acts randomly.
"""

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # `python -m scripts.watch_football_3d`
    from .football_policy import make_policy
except ImportError:  # `python scripts/watch_football_3d.py`
    from football_policy import make_policy

import numpy as np  # noqa: E402  (after sys.path setup below)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponent", default="static", choices=["static", "scripted", "self_play"]
    )
    parser.add_argument("--n-per-team", type=int, default=1)
    parser.add_argument(
        "--formation-mode", default="fixed", choices=["fixed", "random"]
    )
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-npz",
        default=None,
        help="Trained policy from train_football_ars.py (runs/<name>/best.npz).",
    )
    parser.add_argument(
        "--score-every",
        type=float,
        default=2.0,
        help="Seconds between console scoreboard lines.",
    )
    args = parser.parse_args()

    from flybody.fly_envs import football_vs

    holder = {}

    def loader():
        env = football_vs(
            opponent_mode=args.opponent,
            n_per_team=args.n_per_team,
            formation_mode=args.formation_mode,
            time_limit=args.time_limit,
            random_state=np.random.RandomState(args.seed),
        )
        holder["env"] = env
        holder["base"] = None
        holder["last_print"] = 0.0
        return env

    def policy(timestep):
        env = holder.get("env")
        if env is None:
            raise RuntimeError("viewer policy called before env was loaded")
        if holder["base"] is None:
            n_act = env.action_spec().shape[0]
            holder["base"] = make_policy(
                n_act, seed=args.seed + 1, policy_npz=args.policy_npz
            )
        now = time.time()
        if now - holder["last_print"] >= args.score_every:
            holder["last_print"] = now
            task = env.task
            score = task.score
            ev_name, ev_side, _ = task.last_event_info
            print(
                f"t={env.physics.time():6.1f}s "
                f"W {score['west']}:{score['east']} E "
                f"ball:{task.possession_side or '-'} "
                f"last:{ev_name}"
                + (f"({ev_side})" if ev_side else "")
                + f" formations:{task.formation[0]}/{task.formation[1]}",
                flush=True,
            )
        return holder["base"](timestep.observation)

    try:
        from dm_control import viewer

        viewer.launch(loader, policy=policy, title="FlyBody Football 3D")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Could not open the 3D viewer: {exc}")
        print("Render to video instead (needs no display):")
        print(
            "  python -m scripts.rollout_football_video "
            f"--opponent {args.opponent} --n-per-team {args.n_per_team} "
            f"--policy-npz {args.policy_npz or 'runs/<name>/best.npz'} "
            "--out videos/football_rollout.mp4"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
