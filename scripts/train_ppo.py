"""Train fly tasks with PPO (torch, CPU-friendly, no TF/JAX/Ray needed).

Parallel rollout workers + minibatch PPO updates with GAE. Works with any of
the six GUI tasks; football_vs has extra field/opponent flags.

Chunkable: run a few iterations per invocation and resume:
    python -m scripts.train_ppo --iters 5 --run-dir runs/ppo1
    python -m scripts.train_ppo --iters 5 --run-dir runs/ppo1 --resume

Speed reference (12-CPU box, 2-fly football): ~45 wall-sec per sim-sec, so
keep --time-limit short (2-3s) and --rollout-steps modest at first.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # `python -m scripts.train_ppo`
    from .ppo_torch import DEFAULT_CFG, run_ppo
except ImportError:  # `python scripts/train_ppo.py`
    from ppo_torch import DEFAULT_CFG, run_ppo


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="football_vs",
                   choices=["walk_on_ball", "walk_imitation",
                            "flight_imitation", "vision_guided_flight",
                            "football_vs", "template_task"])
    p.add_argument("--run-dir", default="runs/ppo1")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=3.0)
    # Football extras.
    p.add_argument("--opponent", default="static",
                   choices=["static", "scripted", "self_play",
                            "role_self_play"],
                   help="FootballVs opponent mode. role_self_play trains "
                   "1 attacker + 1 goalkeeper per team (forces --n-per-team 2).")
    p.add_argument("--n-per-team", type=int, default=1)
    p.add_argument("--formation-mode", default="fixed",
                   choices=["fixed", "random"])
    p.add_argument("--ball-start", type=float, nargs=2, default=[0.5, 0.0])
    p.add_argument("--attacker-start", type=float, nargs=2, default=[-0.8, 0.0])
    # PPO hyperparameters.
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=512,
                   help="Control steps collected per worker per iteration.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--policy-sizes", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--critic-sizes", type=int, nargs="+", default=[512, 512, 256])
    p.add_argument("--init-log-std", type=float, default=-0.5)
    return p


def env_desc_from_args(args):
    if args.task == "football_vs":
        if args.opponent == "role_self_play" and args.n_per_team != 2:
            raise ValueError("role_self_play trains 1 attacker + 1 goalkeeper "
                             "per team, so --n-per-team must be 2")
        kwargs = dict(
            opponent_mode=args.opponent,
            n_per_team=args.n_per_team,
            formation_mode=args.formation_mode,
            time_limit=args.time_limit,
            ball_start=tuple(args.ball_start),
            attacker_spawn=tuple(args.attacker_start),
        )
    elif args.task == "walk_on_ball":
        kwargs = dict(time_limit=args.time_limit)
    elif args.task == "template_task":
        kwargs = dict(time_limit=args.time_limit)
    else:
        kwargs = dict(time_limit=args.time_limit)
    return {"task": args.task, "kwargs": kwargs}


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = dict(DEFAULT_CFG)
    cfg.update(
        workers=args.workers,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_grad_norm=args.max_grad_norm,
        policy_sizes=tuple(args.policy_sizes),
        critic_sizes=tuple(args.critic_sizes),
        init_log_std=args.init_log_std,
        seed=args.seed,
    )
    print(f"task={args.task} run_dir={args.run_dir} cfg={cfg}", flush=True)
    run_ppo(env_desc_from_args(args), cfg, run_dir=args.run_dir,
            resume=args.resume, max_iters=args.iters)


if __name__ == "__main__":
    main()
