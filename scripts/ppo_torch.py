"""PPO (Proximal Policy Optimization) core for flybody tasks, torch CPU.

CPU-friendly: no TF/JAX/Ray needed. Parallel rollout collection over worker
processes (spawn-safe), minibatch PPO updates with GAE on the main process.

Two entry points:
  - ``run_ppo(env_desc, cfg, run_dir, resume, max_iters)``: programmatic API
    used by the GUI-generated driver scripts. ``env_desc`` is a plain
    picklable dict ``{'task': <name>, 'kwargs': {...}}`` built by
    ``build_env`` below (mirrors the six GUI tasks).
  - ``python -m scripts.train_ppo ...``: CLI wrapper (see train_ppo.py).

Chunkable: pass ``max_iters`` per invocation and ``resume=True`` to continue:
    python -m scripts.train_ppo --iters 5 --run-dir runs/ppo1
    python -m scripts.train_ppo --iters 5 --run-dir runs/ppo1 --resume
"""

import csv
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ppo_torch needs PyTorch (CPU is fine): pip install torch --index-url "
        "https://download.pytorch.org/whl/cpu"
    ) from exc


# ----------------------------------------------------------------------------
# Observations / env construction
# ----------------------------------------------------------------------------

def flatten_obs(observation, keys):
    """Flatten + concat obs dict in fixed (sorted-key) order, float32."""
    return np.concatenate(
        [np.ravel(observation[k]) for k in keys]).astype(np.float32)


def build_env(desc):
    """Build a fly task env from a picklable ``{'task', 'kwargs'}`` dict."""
    task = desc["task"]
    kwargs = dict(desc.get("kwargs", {}))
    if task == "walk_on_ball":
        from flybody.fly_envs import walk_on_ball
        return walk_on_ball(**kwargs)
    if task == "walk_imitation":
        from flybody.fly_envs import walk_imitation
        return walk_imitation(**kwargs)
    if task == "flight_imitation":
        from flybody.fly_envs import flight_imitation
        return flight_imitation(**kwargs)
    if task == "vision_guided_flight":
        from flybody.fly_envs import vision_guided_flight
        return vision_guided_flight(**kwargs)
    if task == "football_vs":
        from flybody.fly_envs import football_vs
        return football_vs(**kwargs)
    if task == "template_task":
        from flybody.fly_envs import template_task
        return template_task(**kwargs)
    raise ValueError(f"unknown PPO task: {task!r}")


# ----------------------------------------------------------------------------
# Actor-critic
# ----------------------------------------------------------------------------

def _mlp(sizes):
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(fan_in, fan_out))
        layers.append(nn.Tanh())
    layers.pop()  # No activation on the output.
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor/critic MLPs, Gaussian + tanh squash to env bounds."""

    def __init__(self, obs_dim, act_dim, policy_sizes, critic_sizes,
                 init_log_std=-0.5):
        super().__init__()
        self.actor = _mlp([obs_dim, *policy_sizes])
        self.mean_head = nn.Linear(policy_sizes[-1], act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(
            torch.full((act_dim,), float(init_log_std)))
        self.critic = _mlp([obs_dim, *critic_sizes])
        self.value_head = nn.Linear(critic_sizes[-1], 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs):
        return self.mean_head(self.actor(obs)), self.value_head(
            self.critic(obs)).squeeze(-1)


def _squash(u, minimum, maximum):
    """Map unbounded u into [minimum, maximum] via tanh (elementwise)."""
    center = (maximum + minimum) / 2.0
    half = (maximum - minimum) / 2.0
    return center + half * torch.tanh(u)


@torch.no_grad()
def act_greedy(actor_critic, obs, minimum, maximum):
    """Deterministic (mean) action, numpy. Used for eval rollouts."""
    actor_critic.eval()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mean, _ = actor_critic(obs_t)
    action = _squash(mean, torch.as_tensor(minimum),
                     torch.as_tensor(maximum))
    return action.squeeze(0).numpy()


# ----------------------------------------------------------------------------
# Parallel rollout workers (spawn-safe: top-level functions only)
# ----------------------------------------------------------------------------

_W = None  # Per-process worker state (see _pool_init below).


def _worker_rollout(payload):
    """Collect ``rollout_steps`` transitions with fixed policy weights.

    Payload: (actor_state, keys, ob_mean, ob_var, minimum, maximum,
    rollout_steps, seed). Returns dict of numpy batches + episode returns.
    """
    (actor_state, keys, ob_mean, ob_var, minimum, maximum, rollout_steps,
     seed) = payload
    import torch as _torch  # Local import: faster spawn on some platforms.

    env, ts = _W["env"], _W["ts"]
    # Per-worker torch seed: otherwise spawn workers share the default RNG
    # state and sample identical action noise within an iteration.
    _torch.manual_seed(seed % (2 ** 32))
    obs_dim = len(ob_mean)
    act_dim = len(minimum)
    policy_sizes, critic_sizes = _W["policy_sizes"], _W["critic_sizes"]
    net = ActorCritic(obs_dim, act_dim, policy_sizes, critic_sizes)
    net.load_state_dict(actor_state)
    net.eval()
    lo = _torch.as_tensor(minimum)
    hi = _torch.as_tensor(maximum)

    obs_b, act_b, logp_b, rew_b, done_b, val_b = [], [], [], [], [], []
    ep_rets, ep_ret = [], 0.0
    obs = flatten_obs(ts.observation, keys)
    for _ in range(rollout_steps):
        norm = ((obs - ob_mean) / np.sqrt(ob_var + 1e-2)).astype(np.float32)
        with _torch.no_grad():
            obs_t = _torch.as_tensor(norm).unsqueeze(0)
            mean, value = net(obs_t)
            std = _torch.exp(net.log_std)
            u = mean + std * _torch.randn_like(mean)
            # Log-prob of sampled u under N(mean, std), with the tanh
            # squash change-of-variables correction.
            uu = u.squeeze(0)
            z = (uu - mean.squeeze(0)) / std
            base = -0.5 * (z ** 2 + 2.0 * net.log_std
                           + np.log(2.0 * np.pi)).sum()
            corr = _torch.log(
                (hi - lo) / 2.0 * (1.0 - _torch.tanh(uu) ** 2)
                + 1e-6).sum()
            logp = (base - corr).unsqueeze(0)
            action = _squash(u, lo, hi).squeeze(0).numpy()
        ts = env.step(action)
        rew = float(ts.reward)
        done = bool(ts.last())
        obs_b.append(obs)
        act_b.append(action)
        logp_b.append(float(logp.item()))
        rew_b.append(rew)
        done_b.append(done)
        val_b.append(float(value.item()))
        ep_ret += rew
        obs = flatten_obs(ts.observation, keys)
        if done:
            ep_rets.append(ep_ret)
            ep_ret = 0.0
    # Bootstrap value for a truncated (non-terminal) chunk end.
    with _torch.no_grad():
        norm = ((obs - ob_mean) / np.sqrt(ob_var + 1e-2)).astype(np.float32)
        _, last_v = net(_torch.as_tensor(norm).unsqueeze(0))
    _W["ts"] = ts
    out = {
        "obs": np.asarray(obs_b, dtype=np.float32),
        "act": np.asarray(act_b, dtype=np.float32),
        "logp": np.asarray(logp_b, dtype=np.float32),
        "rew": np.asarray(rew_b, dtype=np.float32),
        "done": np.asarray(done_b, dtype=bool),
        "val": np.asarray(val_b, dtype=np.float32),
        "last_val": float(last_v.item()),
        "last_done": bool(ts.last()),
        "ep_rets": np.asarray(ep_rets, dtype=np.float32),
    }
    return out


def _worker_eval_greedy(payload):
    """One greedy (mean-action) episode. Returns (return, steps)."""
    (actor_state, keys, ob_mean, ob_var, minimum, maximum,
     policy_sizes, critic_sizes, seed) = payload
    import torch as _torch

    env = _W["env"]  # Shared pool env; reset per eval episode.
    del seed  # Env drives its own randomization.
    obs_dim = len(ob_mean)
    net = ActorCritic(obs_dim, len(minimum), policy_sizes, critic_sizes)
    net.load_state_dict(actor_state)
    net.eval()
    lo = _torch.as_tensor(minimum)
    hi = _torch.as_tensor(maximum)
    ts = env.reset()
    total, steps = 0.0, 0
    while not ts.last():
        obs = flatten_obs(ts.observation, keys)
        norm = ((obs - ob_mean) / np.sqrt(ob_var + 1e-2)).astype(np.float32)
        with _torch.no_grad():
            mean, _ = net(_torch.as_tensor(norm).unsqueeze(0))
            action = _squash(mean, lo, hi).squeeze(0).numpy()
        ts = env.step(action)
        total += float(ts.reward)
        steps += 1
    return total, steps


# ----------------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------------

def _gae(rewards, dones, values, last_val, last_done, gamma, lam):
    adv = np.zeros_like(rewards)
    gae = 0.0
    next_v = 0.0 if last_done else last_val
    for t in reversed(range(len(rewards))):
        nonterm = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_v = values[t]
    return adv


def ppo_update(net, opt_actor, opt_critic, batch, cfg, minimum, maximum,
               log_rates=True):
    """Minibatch PPO epochs over one rollout batch. Returns stat dict."""
    obs = torch.as_tensor(batch["obs"])
    act = torch.as_tensor(batch["act"])
    old_logp = torch.as_tensor(batch["logp"])
    old_val = torch.as_tensor(batch["val"])
    ret = torch.as_tensor(batch["ret"])
    adv = torch.as_tensor(batch["adv"])
    lo = torch.as_tensor(minimum)
    hi = torch.as_tensor(maximum)
    n = len(obs)
    mb = int(cfg.get("minibatch_size", 512))
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clip_frac": 0.0}
    nb = 0
    for _ in range(int(cfg.get("epochs", 3))):
        idx = torch.randperm(n)
        for start in range(0, n, mb):
            sel = idx[start:start + mb]
            mean, value = net(obs[sel])
            std = torch.exp(net.log_std)
            # Log-prob of the stored (squashed) actions: invert the squash.
            clipped = torch.clamp((act[sel] - (hi + lo) / 2.0)
                                  / ((hi - lo) / 2.0 + 1e-8), -1 + 1e-6,
                                  1 - 1e-6)
            u = torch.atanh(clipped)
            z = (u - mean) / std
            base = -0.5 * (z ** 2 + 2.0 * net.log_std
                           + np.log(2.0 * np.pi)).sum(-1)
            corr = torch.log((hi - lo) / 2.0 * (1.0 - torch.tanh(u) ** 2)
                             + 1e-6).sum(-1)
            logp = base - corr
            ratio = torch.exp(logp - old_logp[sel])
            eps = float(cfg.get("clip_eps", 0.2))
            pg1 = ratio * adv[sel]
            pg2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv[sel]
            pg_loss = -torch.min(pg1, pg2).mean()
            v_clipped = old_val[sel] + torch.clamp(
                value - old_val[sel], -eps, eps)
            vf1 = (value - ret[sel]) ** 2
            vf2 = (v_clipped - ret[sel]) ** 2
            vf_loss = torch.max(vf1, vf2).mean()
            entropy = (0.5 * (torch.log(
                2.0 * np.pi * np.e * std ** 2)).sum()
                       - corr.mean())
            loss = (pg_loss
                    + float(cfg.get("value_coef", 0.5)) * vf_loss
                    - float(cfg.get("entropy_coef", 0.01)) * entropy)
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            loss.backward()
            max_norm = float(cfg.get("max_grad_norm", 0.5))
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            opt_actor.step()
            opt_critic.step()
            with torch.no_grad():
                kl = (old_logp[sel] - logp).mean()
                clip_frac = ((ratio - 1.0).abs() > eps).float().mean()
            stats["pg"] += float(pg_loss)
            stats["vf"] += float(vf_loss)
            stats["ent"] += float(entropy)
            stats["kl"] += float(kl)
            stats["clip_frac"] += float(clip_frac)
            nb += 1
    if log_rates and nb:
        for k in stats:
            stats[k] /= nb
    return stats


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

DEFAULT_CFG = {
    "workers": 8,
    "rollout_steps": 512,
    "epochs": 3,
    "minibatch_size": 512,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "max_grad_norm": 0.5,
    "policy_sizes": (256, 256, 256),
    "critic_sizes": (512, 512, 256),
    "init_log_std": -0.5,
    "seed": 0,
}


def run_ppo(env_desc, cfg=None, run_dir="runs/ppo1", resume=False,
            max_iters=None):
    """Train PPO on ``env_desc``. Chunkable via ``max_iters`` + ``resume``."""
    import concurrent.futures as futures

    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(run_dir, "latest.pt")
    best_path = os.path.join(run_dir, "best.pt")
    log_path = os.path.join(run_dir, "log.csv")

    # Probe one env for shapes / keys / bounds.
    probe = build_env(env_desc)
    ts = probe.reset()
    keys = sorted(ts.observation.keys())
    obs_dim = len(flatten_obs(ts.observation, keys))
    spec = probe.action_spec()
    act_dim = spec.shape[0]
    minimum = np.asarray(spec.minimum, dtype=np.float32)
    maximum = np.asarray(spec.maximum, dtype=np.float32)
    del probe

    policy_sizes = tuple(cfg["policy_sizes"])
    critic_sizes = tuple(cfg["critic_sizes"])
    net = ActorCritic(obs_dim, act_dim, policy_sizes, critic_sizes,
                      cfg["init_log_std"])
    opt_actor = torch.optim.Adam(
        list(net.actor.parameters()) + list(net.mean_head.parameters())
        + [net.log_std], lr=float(cfg["actor_lr"]))
    opt_critic = torch.optim.Adam(
        list(net.critic.parameters()) + list(net.value_head.parameters()),
        lr=float(cfg["critic_lr"]))

    if resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["net"])
        opt_actor.load_state_dict(ckpt["opt_actor"])
        opt_critic.load_state_dict(ckpt["opt_critic"])
        ob_mean, ob_var = ckpt["ob_mean"], ckpt["ob_var"]
        best = float(ckpt["best"])
        start_iter = int(ckpt["iter"]) + 1
        print(f"resumed at iter {start_iter}, best_eval={best:.2f}")
    else:
        ob_mean = np.zeros(obs_dim, dtype=np.float32)
        ob_var = np.ones(obs_dim, dtype=np.float32)
        best = -1e18
        start_iter = 0
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["iter", "mean_ret", "eval_ret", "pg_loss", "v_loss",
                 "entropy", "kl", "clip_frac"])

    it = start_iter
    with futures.ProcessPoolExecutor(
            max_workers=int(cfg["workers"]),
            initializer=_pool_init,
            initargs=(env_desc, policy_sizes, critic_sizes)) as pool:
        while True:
            if max_iters is not None and it >= start_iter + max_iters:
                break
            actor_state = net.state_dict()
            payloads = [
                (actor_state, keys, ob_mean, ob_var, minimum, maximum,
                 int(cfg["rollout_steps"]),
                 int(cfg["seed"]) * 100000 + it * 1000 + w)
                for w in range(int(cfg["workers"]))
            ]
            chunks = list(pool.map(_worker_rollout, payloads))

            # Merge + GAE per worker chunk.
            obs_all, act_all, logp_all = [], [], []
            ret_all, adv_all, val_all = [], [], []
            ep_rets_all = []
            obs_stat_sum = np.zeros_like(ob_mean, dtype=np.float64)
            obs_stat_sq = np.zeros_like(ob_mean, dtype=np.float64)
            obs_stat_n = 0
            for ch in chunks:
                adv = _gae(ch["rew"], ch["done"], ch["val"], ch["last_val"],
                           ch["last_done"], float(cfg["gamma"]),
                           float(cfg["gae_lambda"]))
                obs_all.append(ch["obs"])
                act_all.append(ch["act"])
                logp_all.append(ch["logp"])
                val_all.append(ch["val"])
                ret_all.append(ch["val"] + adv)
                adv_all.append(adv)
                ep_rets_all.append(ch["ep_rets"])
                obs_stat_sum += ch["obs"].astype(np.float64).sum(0)
                obs_stat_sq += (ch["obs"].astype(np.float64) ** 2).sum(0)
                obs_stat_n += len(ch["obs"])
            obs_all = np.concatenate(obs_all)
            act_all = np.concatenate(act_all)
            logp_all = np.concatenate(logp_all)
            val_all = np.concatenate(val_all)
            ret_all = np.concatenate(ret_all)
            adv_all = np.concatenate(adv_all)
            adv_all = (adv_all - adv_all.mean()
                       / max(1e-8, adv_all.std() + 1e-8)).astype(np.float32)
            ep_rets = (np.concatenate(ep_rets_all)
                       if any(len(e) for e in ep_rets_all)
                       else np.zeros(0, dtype=np.float32))
            mean_ret = float(ep_rets.mean()) if len(ep_rets) else float("nan")

            batch = {"obs": ((obs_all - ob_mean)
                             / np.sqrt(ob_var + 1e-2)).astype(np.float32),
                     "act": act_all, "logp": logp_all, "val": val_all,
                     "ret": ret_all, "adv": adv_all}
            stats = ppo_update(net, opt_actor, opt_critic, batch, cfg,
                               minimum, maximum)

            # Refresh obs normalizer from this iteration's states.
            ob_mean = (obs_stat_sum / max(1, obs_stat_n)).astype(np.float32)
            sq = obs_stat_sq / max(1, obs_stat_n)
            ob_var = np.maximum(
                sq - ob_mean.astype(np.float64) ** 2,
                1e-2).astype(np.float32)

            # Greedy eval episode (drives best.pt).
            eval_ret, _ = _eval_greedy(
                pool, net, keys, ob_mean, ob_var, minimum, maximum,
                policy_sizes, critic_sizes, int(cfg["seed"]) + it)
            if eval_ret > best:
                best = eval_ret
                torch.save({
                    "net": net.state_dict(), "ob_mean": ob_mean,
                    "ob_var": ob_var, "keys": np.array(keys),
                    "minimum": minimum, "maximum": maximum,
                    "policy_sizes": policy_sizes,
                    "critic_sizes": critic_sizes,
                    "eval_ret": best, "iter": it,
                }, best_path)
            torch.save({
                "net": net.state_dict(),
                "opt_actor": opt_actor.state_dict(),
                "opt_critic": opt_critic.state_dict(),
                "ob_mean": ob_mean, "ob_var": ob_var, "best": best,
                "iter": it,
            }, latest)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [it, f"{mean_ret:.3f}", f"{eval_ret:.3f}",
                     f"{stats['pg']:.4f}", f"{stats['vf']:.4f}",
                     f"{stats['ent']:.4f}", f"{stats['kl']:.4f}",
                     f"{stats['clip_frac']:.3f}"])
            print(f"iter {it}: mean_ret={mean_ret:.2f} "
                  f"eval_ret={eval_ret:.2f} best={best:.2f} "
                  f"kl={stats['kl']:.4f}", flush=True)
            it += 1
    print(f"done. best_eval={best:.2f} -> {best_path}")
    return best


def _pool_init(env_desc, policy_sizes, critic_sizes):
    global _W
    torch.set_num_threads(1)
    env = build_env(env_desc)
    spec = env.action_spec()
    _W = {
        "env": env,
        "minimum": np.asarray(spec.minimum, dtype=np.float32),
        "maximum": np.asarray(spec.maximum, dtype=np.float32),
        "ts": env.reset(),
        "policy_sizes": tuple(policy_sizes),
        "critic_sizes": tuple(critic_sizes),
    }


def _eval_greedy(pool, net, keys, ob_mean, ob_var, minimum, maximum,
                 policy_sizes, critic_sizes, seed):
    payload = (net.state_dict(), keys, ob_mean, ob_var, minimum, maximum,
               tuple(policy_sizes), tuple(critic_sizes), seed)
    rets = list(pool.map(_worker_eval_greedy, [payload]))
    return float(np.mean([r[0] for r in rets])), int(sum(r[1] for r in rets))


def load_greedy_policy(path):
    """Load best.pt -> numpy greedy policy fn + meta (for custom rollouts)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = [str(k) for k in ckpt["keys"]]
    minimum = np.asarray(ckpt["minimum"], dtype=np.float32)
    maximum = np.asarray(ckpt["maximum"], dtype=np.float32)
    net = ActorCritic(len(ckpt["ob_mean"]), len(minimum),
                      tuple(ckpt["policy_sizes"]),
                      tuple(ckpt["critic_sizes"]))
    net.load_state_dict(ckpt["net"])
    net.eval()
    ob_mean, ob_var = ckpt["ob_mean"], ckpt["ob_var"]

    def policy(observation, _keys=keys):
        flat = flatten_obs(observation, _keys)
        norm = ((flat - ob_mean) / np.sqrt(ob_var + 1e-2)).astype(np.float32)
        return act_greedy(net, norm, minimum, maximum)

    return policy, {"keys": keys, "eval_ret": ckpt.get("eval_ret")}
