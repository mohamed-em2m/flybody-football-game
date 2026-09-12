"""Shared linear-policy helpers for football_vs training.

The observation flattening order (sorted keys) must stay identical between
training and rollout, so both import from here.
"""

import os
import numpy as np


def flatten_obs(observation, keys):
    return np.concatenate([np.ravel(observation[k]) for k in keys]).astype(float)


def linear_action(flat_obs, matrix, mean, var, minimum, maximum, eps=1e-2):
    normed = (flat_obs - mean) / np.sqrt(var + eps)
    if normed.ndim == 1:
        action = matrix @ normed
    else:
        action = normed @ matrix.T
    return np.clip(action, minimum, maximum)


def save_policy(path, matrix, mean, var, keys, minimum, maximum):
    np.savez(
        path,
        matrix=matrix,
        mean=mean,
        var=var,
        keys=np.array(keys),
        minimum=minimum,
        maximum=maximum,
    )


def load_policy(path):
    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path = path + ".npz"
    with np.load(path, allow_pickle=True) as data:
        matrix = np.array(data["matrix"])
        mean = np.array(data["mean"])
        var = np.array(data["var"])
        keys = [
            k.decode("utf-8") if isinstance(k, (bytes, np.bytes_)) else str(k)
            for k in data["keys"]
        ]
        minimum = np.array(data["minimum"])
        maximum = np.array(data["maximum"])
    return matrix, mean, var, keys, minimum, maximum


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
