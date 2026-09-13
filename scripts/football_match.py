"""Top-down match director + kinematic 3D playback for football_vs.

Runs the match brain (chase, pass, shoot, counter, goals) in 2D and
replays it through the MuJoCo scene: fly bodies are placed kinematically
each step with procedurally stepping legs, the ball flies director-side.
Kickoff layouts come from the env's own formation system, so the video
shows the same shapes training sees. Zero training required.

Pipeline (see rollout_football_video --mode match):
  director = MatchDirector(task, seed)   # reads task._spawns
  director.step(dt)                      # 2D brain
  action = make_frame_action(...)        # legs for every fly
  apply_kinematic(...)                   # teleport bodies + ball
  env.step(action); render
"""

import numpy as np

try:  # `python -m ...` package usage
    from .football_scripted import leg_vector
except ImportError:  # direct script usage
    from football_scripted import leg_vector

_CHASE_SPEED = 0.62
_SUPPORT_SPEED = 0.5
_TURN_RATE = 7.0
_TOUCH_DIST = 0.30
_SEP_DIST = 0.25
_KICK_CD = 0.4


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class MatchDirector:
    """2D football brain. Units are arena meters, dt in seconds."""

    def __init__(self, task, seed=0, pace=1.0):
        self._task = task
        self._rng = np.random.RandomState(seed)
        self._pace = pace
        self._L = task._arena.field_length
        self._Wd = task._arena.field_width
        self._gx_e = task._arena.east_goal_x
        self._gx_w = task._arena.west_goal_x
        self._mouth = task._arena.goal_width / 2.0
        self._ball_r = task._arena.ball_radius
        self._west = [w.name for w in task._west]
        self._east = [w.name for w in task._east]
        self.score = {"west": 0, "east": 0}
        self.last_event = ("none", None)
        self.reset_kickoff()

    def reset_kickoff(self):
        spawns = self._task._spawns
        self.pos = {n: [float(s[0]), float(s[1])] for n, s in spawns.items()}
        self.prev = {n: list(p) for n, p in self.pos.items()}
        self.yaw = {}
        for n in self._west:
            self.yaw[n] = 0.0
        for n in self._east:
            self.yaw[n] = np.pi
        self.prev_yaw = dict(self.yaw)
        self.ball = [0.0 + self._rng.uniform(-0.1, 0.1), self._rng.uniform(-0.1, 0.1)]
        self.bvel = [0.0, 0.0]
        self.kick_cd = 0.0
        self.poss = None
        self.phases = {n: self._rng.uniform(0, 2 * np.pi) for n in self.pos}

    def _rank(self, names):
        bx, by = self.ball
        return sorted(names, key=lambda n: (self.pos[n][0] - bx) ** 2 + (self.pos[n][1] - by) ** 2)

    def _separate(self):
        for a in (self._west, self._east):
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    u, v = a[i], a[j]
                    sx = self.pos[u][0] - self.pos[v][0]
                    sy = self.pos[u][1] - self.pos[v][1]
                    d = float(np.hypot(sx, sy))
                    if 1e-6 < d < _SEP_DIST:
                        push = (_SEP_DIST - d) / 2.0
                        ux, uy = sx / d, sy / d
                        self.pos[u][0] += ux * push
                        self.pos[u][1] += uy * push
                        self.pos[v][0] -= ux * push
                        self.pos[v][1] -= uy * push

    def _move(self, name, tx, ty, speed, dt):
        x, y = self.pos[name]
        want = np.arctan2(ty - y, tx - x)
        turn = _wrap(want - self.yaw[name])
        mx = _TURN_RATE * dt
        turn = max(-mx, min(mx, turn))
        self.yaw[name] = self.yaw[name] + turn
        step = speed * self._pace * dt
        self.pos[name][0] = x + np.cos(self.yaw[name]) * step
        self.pos[name][1] = y + np.sin(self.yaw[name]) * step
        self.phases[name] += 2.0 * np.pi * 6.0 * dt * (0.3 + speed)

    @staticmethod
    def _tag(name):
        return name[-1] if name[-1].isdigit() else name[:3]

    def _team_kick(self, name, own, foes, goal_x, side):
        bx, by = self.ball

        def foe_d(p):
            return min(
                float(np.hypot(self.pos[d][0] - p[0], self.pos[d][1] - p[1]))
                for d in foes
            )

        press = foe_d(self.pos[name])
        best, best_open = None, -1e9
        for m in [m for m in own if m != name]:
            gap = foe_d(self.pos[m]) - press
            if gap > best_open:
                best_open, best = gap, m
        dist_goal = abs(goal_x - bx)
        r = self._rng.uniform()
        tag = self._tag(name)
        was_foe = self.poss is not None and self.poss != side
        if dist_goal < 1.2 and r < 0.8:
            tx = goal_x
            ty = self._rng.uniform(-0.7, 0.7) * self._mouth
            sp = self._rng.uniform(2.0, 2.6)
            ev = f"shot {tag}"
        elif best is not None and best_open > (0.15 if press < 0.4 else 0.3) and r < 0.75:
            tx, ty = self.pos[best][0], self.pos[best][1]
            sp = self._rng.uniform(1.8, 2.4)
            ev = f"pass {tag}->{self._tag(best)}"
        elif was_foe:
            # Counter-attack: won it off them, hoof it straight upfield.
            tx = goal_x
            ty = self._rng.uniform(-0.8, 0.8) * self._mouth
            sp = self._rng.uniform(2.4, 3.0)
            ev = f"counter {tag}"
        else:
            # Dribble: soft touch ahead toward goal; keeps the rally alive.
            tx = goal_x
            ty = by + self._rng.uniform(-0.5, 0.5)
            sp = self._rng.uniform(0.9, 1.3)
            ev = f"dribble {tag}"
        a = np.arctan2(ty - by, tx - bx)
        self.bvel = [np.cos(a) * sp, np.sin(a) * sp]
        self.kick_cd = _KICK_CD
        self.poss = side
        self.last_event = (ev, side)

    def _west_kick(self, name):
        self._team_kick(name, self._west, self._east, self._gx_e, "west")

    def _east_kick(self, name):
        self._team_kick(name, self._east, self._west, self._gx_w, "east")

    def step(self, dt):
        """Advance the match. Returns 'west_goal'/'east_goal'/None."""
        self.kick_cd = max(0.0, self.kick_cd - dt)
        bx, by = self.ball
        # West: two nearest chase, rest support ahead + spread.
        wrank = self._rank(self._west)
        for k, name in enumerate(wrank[:2]):
            lead = 0.25 if (self.bvel[0] ** 2 + self.bvel[1] ** 2) ** 0.5 > 1.0 else 0.0
            px = bx + self.bvel[0] * lead + (0.16 if k == 1 else 0.0)
            py = by + self.bvel[1] * lead + (0.2 if k == 1 else 0.0) * (1 if k % 2 else -1)
            self._move(name, px, py, _CHASE_SPEED, dt)
        for k, name in enumerate(wrank[2:]):
            tx = min(bx + 0.8, self._gx_e - 0.4)
            ty = (k - (len(wrank[2:]) - 1) / 2.0) * 0.5
            self._move(name, tx, ty, _SUPPORT_SPEED, dt)
        # East: nearest presses, rest contain goalside.
        erank = self._rank(self._east)
        for k, name in enumerate(erank):
            txx = bx + (self._gx_e - bx) * 0.10 * k
            tyy = by + (0.0 - by) * 0.05 * k
            self._move(name, txx, tyy, _CHASE_SPEED, dt)
        self._separate()
        # Contacts.
        if self.kick_cd <= 0:
            for name in wrank + erank:
                d = float(np.hypot(self.pos[name][0] - bx, self.pos[name][1] - by))
                if d <= _TOUCH_DIST:
                    if name in self._west:
                        self._west_kick(name)
                    else:
                        self._east_kick(name)
                    break
        # Ball.
        fr = 0.4 ** dt  # halves roughly every 1.2 s
        self.bvel[0] *= fr
        self.bvel[1] *= fr
        bx += self.bvel[0] * dt
        by += self.bvel[1] * dt
        # Long-side walls (indoor).
        edge = self._Wd / 2.0 - self._ball_r
        if abs(by) > edge:
            by = np.sign(by) * edge
            self.bvel[1] *= -0.6
        # Short sides: mouth = goal, else bounce.
        if bx > self._gx_e and abs(by) < self._mouth:
            self.score["west"] += 1
            self.last_event = ("GOAL west", "west")
            return "west_goal"
        if bx < self._gx_w and abs(by) < self._mouth:
            self.score["east"] += 1
            self.last_event = ("GOAL east", "east")
            return "east_goal"
        if bx > self._gx_e + 0.4 or bx < self._gx_w - 0.4:
            self.bvel[0] *= -0.6
            bx = min(max(bx, self._gx_w - 0.4), self._gx_e + 0.4)
        self.ball = [bx, by]
        # Keep flies near the pitch (soft arena walls for the bodies).
        for names in (self._west, self._east):
            for name in names:
                self.pos[name][0] = min(
                    max(self.pos[name][0], -self._L / 2.0 - 0.3), self._L / 2.0 + 0.3
                )
                self.pos[name][1] = min(
                    max(self.pos[name][1], -self._Wd / 2.0 - 0.3), self._Wd / 2.0 + 0.3
                )
        return None


def build_leg_maps(task):
    """{fly_name: ({actuator: action idx}, per-fly dim)} for animation."""
    maps = {}
    for w in task._fly_order:
        local = [a.name for a in w.mjcf_model.find_all("actuator")]
        pos = {}
        for ap, cl in zip(w._action_indices["legs"], w._ctrl_indices["legs"]):
            pos[local[cl]] = ap
        dim = sum(len(v) for v in w._action_indices.values())
        maps[w.name] = (pos, dim)
    return maps


def advance_legs(director, dt, freq=6.0):
    """Advance every fly's gait phase by a render-frame dt."""
    for name in director.phases:
        x, y = director.pos[name]
        px, py = director.prev[name]
        spd = float(np.hypot(x - px, y - py)) / max(dt, 1e-6)
        director.phases[name] += 2.0 * np.pi * freq * dt * (0.25 + min(1.0, spd))


def apply_frame(physics, task, director, leg_maps, dt, spawn_z):
    """Place bodies + ball from director state; pose stepping legs.

    Legs are written straight to ctrl (playback needs no env stepping).
    """
    for name in director.pos:
        x, y = director.pos[name]
        px, py = director.prev[name]
        yaw = director.yaw[name]
        pyaw = director.prev_yaw[name]
        physics.bind(task._root_joints[name]).qpos = np.array(
            [x, y, spawn_z, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        )
        physics.bind(task._root_joints[name]).qvel = np.array(
            [
                (x - px) / dt,
                (y - py) / dt,
                0.0,
                0.0,
                0.0,
                _wrap(yaw - pyaw) / dt,
            ]
        )
        director.prev[name] = [x, y]
        director.prev_yaw[name] = yaw
        pos, dim = leg_maps[name]
        spd = float(np.hypot(x - px, y - py)) / max(dt, 1e-6)
        stride = min(0.6, 0.15 + spd * 0.5)
        vec = leg_vector(pos, dim, director.phases[name], stride, stride, 0.35)
        rev = {ap: s for s, ap in pos.items()}
        for ap in range(dim):
            short = rev.get(ap)
            if short is None:
                continue
            try:
                physics.named.data.ctrl[f"{name}/{short}"] = vec[ap]
            except KeyError:
                pass
    bx, by = director.ball
    r = task._arena.ball_radius
    physics.named.data.qpos["football"] = np.array(
        [bx, by, r + 0.01, 1.0, 0.0, 0.0, 0.0]
    )
    physics.named.data.qvel["football"] = np.array(
        [director.bvel[0], director.bvel[1], 0.0, 0.0, 0.0, 0.0]
    )


def draw_match_scoreboard(frame, director):
    """PIL overlay driven by the director (mirrors the video overlay)."""
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.rectangle([8, 8, 360, 78], fill=(0, 0, 0))
        d.text(
            (14, 12),
            f"W {director.score['west']} : {director.score['east']} E",
            fill=(255, 255, 255),
        )
        ev, side = director.last_event
        d.text(
            (14, 28),
            f"ball: {director.poss or '-'}   last: {ev}" + (f" ({side})" if side else ""),
            fill=(200, 200, 200),
        )
        d.text(
            (14, 44),
            "W: " + " ".join(director._west) + "  E: " + " ".join(director._east),
            fill=(255, 255, 255),
        )
        return np.asarray(img)
    except Exception:
        return frame
