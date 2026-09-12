"""Train football_vs with Augmented Random Search (ARS) on a linear policy.

CPU-friendly: no TF/JAX/Ray needed. Parallel rollouts over worker processes.

Chunkable: run a few iterations per invocation and resume:
    python train_football_ars.py --iters 3 --run-dir runs/ars1
    python train_football_ars.py --iters 3 --run-dir runs/ars1 --resume

Speed reference (12-CPU box, 2 flies): ~45 wall-sec per sim-sec, so a
3s episode takes ~2min; with 8 workers x 8 mirrored dirs one iteration
takes ~5min.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from .football_policy import flatten_obs, linear_action, save_policy

_W = None  # Per-process worker state.


def _worker_init(env_kwargs):
    global _W
    from flybody.fly_envs import football_vs

    env = football_vs(**env_kwargs)
    spec = env.action_spec()
    _W = {
        "env": env,
        "minimum": np.asarray(spec.minimum, dtype=float),
        "maximum": np.asarray(spec.maximum, dtype=float),
    }


def _worker_eval(payload):
    """Evaluate one perturbed policy. Returns (return, sum, sqsum, n)."""
    matrix, mean, var, keys, delta, sign, noise, seed = payload
    env, minimum, maximum = _W["env"], _W["minimum"], _W["maximum"]
    theta = matrix + sign * noise * delta
    ts = env.reset()
    total = 0.0
    obs_sum = np.zeros_like(mean)
    obs_sqsum = np.zeros_like(mean)
    n = 0
    rng = np.random.RandomState(seed)
    del rng  # Env drives its own randomization.
    while not ts.last():
        flat = flatten_obs(ts.observation, keys)
        obs_sum += flat
        obs_sqsum += flat * flat
        n += 1
        action = linear_action(flat, theta, mean, var, minimum, maximum)
        ts = env.step(action)
        total += float(ts.reward)
    return total, obs_sum, obs_sqsum, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/ars1")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--dirs",
        type=int,
        default=8,
        help="Perturbation directions per iter (x2 mirrored).",
    )
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument(
        "--ball-start",
        type=float,
        nargs=2,
        default=[0.5, 0.0],
        help="Curriculum ball start. Near goal = easy.",
    )
    parser.add_argument("--attacker-start", type=float, nargs=2, default=[-0.8, 0.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    latest = os.path.join(args.run_dir, "latest.npz")
    best_path = os.path.join(args.run_dir, "best.npz")
    log_path = os.path.join(args.run_dir, "log.csv")

    from flybody.fly_envs import football_vs  # noqa: F401  (warms import)
    import concurrent.futures as futures

    # Probe one env for shapes / keys / bounds.
    from flybody.fly_envs import football_vs as make_env

    probe = make_env(opponent_mode="static", time_limit=args.time_limit)
    ts = probe.reset()
    keys = sorted(ts.observation.keys())
    obs_dim = len(flatten_obs(ts.observation, keys))
    spec = probe.action_spec()
    act_dim = spec.shape[0]
    minimum = np.asarray(spec.minimum, dtype=float)
    maximum = np.asarray(spec.maximum, dtype=float)
    del probe

    rng = np.random.RandomState(args.seed)
    if args.resume and os.path.exists(latest):
        data = np.load(latest, allow_pickle=True)
        matrix = data["matrix"]
        mean, var = data["mean"], data["var"]
        best = float(data["best"])
        start_iter = int(data["iter"]) + 1
        print(f"resumed at iter {start_iter}, best={best:.2f}")
    else:
        matrix = np.zeros((act_dim, obs_dim))
        mean = np.zeros(obs_dim)
        var = np.ones(obs_dim)
        best = -1e18
        start_iter = 0
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "mean_ret", "best_ret", "sigma_r"])

    env_kwargs = dict(
        opponent_mode="static",
        time_limit=args.time_limit,
        ball_start=tuple(args.ball_start),
        attacker_spawn=tuple(args.attacker_start),
    )
    with futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init, initargs=(env_kwargs,)
    ) as pool:
        for it in range(start_iter, start_iter + args.iters):
            deltas = [rng.randn(*matrix.shape) for _ in range(args.dirs)]
            payloads = []
            for i, d in enumerate(deltas):
                for sign in (+1.0, -1.0):
                    payloads.append(
                        (
                            matrix,
                            mean,
                            var,
                            keys,
                            d,
                            sign,
                            args.noise,
                            args.seed * 100000 + it * 1000 + i,
                        )
                    )
            results = list(pool.map(_worker_eval, payloads))

            rets = np.array([r[0] for r in results]).reshape(args.dirs, 2)
            # Update obs normalizer with this iteration's states.
            total_n = sum(r[3] for r in results)
            mean = sum(r[1] for r in results) / total_n
            sq = sum(r[2] for r in results) / total_n
            var = np.maximum(sq - mean * mean, 1e-2)

            # ARS V2 update on top-k directions.
            order = np.argsort(rets.max(axis=1))[::-1][: args.top]
            sigma_r = rets[order].std() + 1e-8
            step = np.zeros_like(matrix)
            for i in order:
                step += (rets[i, 0] - rets[i, 1]) * deltas[i]
            matrix += (args.step_size / (args.top * sigma_r)) * step

            mean_ret = float(rets.mean())
            best_ret = float(rets.max())
            if best_ret > best:
                best = best_ret
                save_policy(best_path, matrix, mean, var, keys, minimum, maximum)
            np.savez(
                latest,
                matrix=matrix,
                mean=mean,
                var=var,
                keys=np.array(keys),
                minimum=minimum,
                maximum=maximum,
                best=best,
                iter=it,
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [it, f"{mean_ret:.3f}", f"{best_ret:.3f}", f"{sigma_r:.3f}"]
                )
            print(
                f"iter {it}: mean={mean_ret:.2f} best_ep={best_ret:.2f} "
                f"best={best:.2f}",
                flush=True,
            )
    print(f"done. best={best:.2f} -> {best_path}")


if __name__ == "__main__":
    main()
