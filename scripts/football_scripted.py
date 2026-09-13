"""Heuristic scripted football policies (no training needed).

Each fly runs a tripod-gait walker: alternating leg tripods step toward
the ball, and the fly lunges when close and facing the opponent goal.
Ball contact does the rest, so matches genuinely play in 3D.

Use via ``make_scripted_match(task)`` which returns ``(policies, full)``:
``policies`` maps fly name -> per-fly callable, and ``full(step,
physics)`` concatenates chunks in team order for
``opponent_mode='self_play'`` (the mode where one action vector drives
every fly, so no env-side policy wiring is needed at all).
"""

import numpy as np

_DT = 0.002  # Control step (s).

_TRIPOD_OFFSETS = {}
for _s, _i in (("left", 1), ("right", 2), ("left", 3)):
    _TRIPOD_OFFSETS[(_s, _i)] = 0.0
for _s, _i in (("right", 1), ("left", 2), ("right", 3)):
    _TRIPOD_OFFSETS[(_s, _i)] = np.pi


def leg_vector(pos, dim, phase, stride_l, stride_r, lift, coxa_bias=0.5):
    """Pure tripod-gait leg command (no physics access).

    Args:
        pos: {actuator short name: action index} for one fly.
        dim: Per-fly action dimension.
        phase: Gait phase (radians); advance ~2*pi*freq*dt per step.
        stride_l/r: Fore-aft sweep amplitude per side (steering).
        lift: Swing lift amplitude. coxa_bias: sweep center (ctrlrange).
    """
    action = np.zeros(dim)
    for (s, i), off in _TRIPOD_OFFSETS.items():
        p = phase + off
        coxa = coxa_bias + (stride_l if s == "left" else stride_r) * np.cos(p)
        femur = lift * max(0.0, np.sin(p))
        action[pos[f"coxa_T{i}_{s}"]] = coxa
        action[pos[f"femur_T{i}_{s}"]] = femur
    return action


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class TripodPolicy:
    """Chase-and-lunge tripod gait for one fly.

    Args:
        task: FootballVs task (for arena goals + root joints).
        fly_name: e.g. 'west_0' / 'east_2'.
        side: 'west' (attacks east goal) or 'east'.
        freq: Stride frequency (Hz).
        stride: Fore-aft coxa sweep amplitude (ctrl units).
        lift: Femur lift amplitude during swing (ctrl units, sign tuned).
        home: Optional (x, y) anchor for goalkeepers. When set, the fly
            holds home and only chases the ball while it is within
            guard_radius of home; otherwise it walks back home.
        guard_radius: Chase radius around home (ignored when home=None).
    """

    def __init__(
        self, task, fly_name, side, freq=6.0, stride=0.6, lift=0.35,
        home=None, guard_radius=1.0,
    ):
        self._task = task
        self._name = fly_name
        self._side = side
        self._freq = freq
        self._stride = stride
        self._lift = lift
        self._coxa_bias = 0.5  # Center sweep inside coxa ctrlrange.
        self._task = task
        self._name = fly_name
        self._side = side
        self._freq = freq
        self._stride = stride
        self._lift = lift
        self._home = None if home is None else (float(home[0]), float(home[1]))
        self._guard_radius = float(guard_radius)
        walker = task._fly_by_name[fly_name]
        local = [a.name for a in walker.mjcf_model.find_all("actuator")]
        self._pos = {}
        a_idx = walker._action_indices["legs"]
        c_idx = walker._ctrl_indices["legs"]
        for ap, cl in zip(a_idx, c_idx):
            self._pos[local[cl]] = ap
        self._dim = sum(len(v) for v in walker._action_indices.values())
        self._phase = 0.0
        self._cool = 0
        self._off = dict(_TRIPOD_OFFSETS)

    def __call__(self, step, physics):
        task = self._task
        rj = physics.bind(task._root_joints[self._name]).qpos
        x, y = float(rj[0]), float(rj[1])
        qw, qx, qy, qz = (float(rj[3]), float(rj[4]), float(rj[5]), float(rj[6]))
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qz * qw)
        yaw = np.arctan2(fy, fx)

        ball = physics.named.data.qpos["football"]
        tx, ty = float(ball[0]), float(ball[1])
        if self._home is not None:
            # Goalkeeper: stay home unless the ball comes near.
            hx, hy = self._home
            if float(np.hypot(tx - hx, ty - hy)) > self._guard_radius:
                tx, ty = hx, hy
        dx, dy = tx - x, ty - y
        dist = float(np.hypot(dx, dy))
        err = _wrap(np.arctan2(dy, dx) - yaw)

        goal_x = (
            task._arena.east_goal_x
            if self._side == "west"
            else task._arena.west_goal_x
        )
        goal_err = _wrap(np.arctan2(0.0 - y, goal_x - x) - yaw)
        lunge = dist < 0.55 and abs(goal_err) < 0.6 and self._cool <= 0

        self._phase += 2.0 * np.pi * self._freq * _DT
        turn = float(np.clip(2.5 * err, -0.9, 0.9))
        surge = 2.0 if lunge else 1.0
        stride_l = self._stride * (1.0 - turn) * surge
        stride_r = self._stride * (1.0 + turn) * surge

        action = leg_vector(
            self._pos,
            self._dim,
            self._phase,
            stride_l,
            stride_r,
            self._lift,
            self._coxa_bias,
        )

        self._cool = max(0, self._cool - 1)
        if lunge:
            self._cool = 150
        return action


def scripted_policies(task, freq=6.0, stride=0.6, lift=0.35):
    """{fly_name: TripodPolicy} driving EVERY fly as a chaser."""
    policies = {}
    for w in task._west:
        policies[w.name] = TripodPolicy(task, w.name, "west", freq, stride, lift)
    for w in task._east:
        policies[w.name] = TripodPolicy(task, w.name, "east", freq, stride, lift)
    return policies


def scripted_role_policies(task, freq=6.0, stride=0.6, lift=0.35,
                           guard_radius=1.0):
    """{fly_name: policy} with attacker chase + goalkeeper home guard.

    Index 0 of each team chases (attacker); index 1 holds its own goal
    mouth and only steps out while the ball is within guard_radius
    (goalkeeper). Falls back to pure chase when the task has no roles.
    """
    roles = getattr(task, "roles", {})
    policies = {}
    for side, flies in (("west", task._west), ("east", task._east)):
        for i, w in enumerate(flies):
            role = roles.get(w.name, "attacker" if i == 0 else "goalkeeper")
            if role == "goalkeeper":
                try:
                    home = task._role_home(side)[:2]
                except Exception:  # pylint: disable=broad-except
                    gx = (task._arena.west_goal_x if side == "west"
                          else task._arena.east_goal_x)
                    home = (gx, 0.0)
                policies[w.name] = TripodPolicy(
                    task, w.name, side, freq, stride, lift,
                    home=home, guard_radius=guard_radius,
                )
            else:
                policies[w.name] = TripodPolicy(
                    task, w.name, side, freq, stride, lift
                )
    return policies


def make_scripted_match(task, freq=6.0, stride=0.6, lift=0.35,
                          use_roles=False, guard_radius=1.0):
    """(policies, full_action) for a self-play-style scripted match.

    ``full_action(step, physics)`` returns the concatenated per-fly action
    vector in ``task._fly_order``, ready for ``opponent_mode='self_play'``
    (or 'role_self_play' with use_roles=True for attacker+goalkeeper
    behavior).
    """
    if use_roles:
        policies = scripted_role_policies(
            task, freq, stride, lift, guard_radius=guard_radius
        )
    else:
        policies = scripted_policies(task, freq, stride, lift)
    order = [w.name for w in task._fly_order]

    def full_action(step, physics):
        return np.concatenate([policies[name](step, physics) for name in order])

    return policies, full_action
