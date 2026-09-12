"""Fly-vs-fly football (soccer) goal-scoring task.

Two teams of flies share one physics and play two-sided football: the WEST
team defends the west (-x) goal and attacks the east (+x) goal, the EAST
team does the opposite. Both sides can score, intercept, pass and shoot.

Modes:
  - opponent_mode='static': west_0 (the attacker) is controlled, every
      other fly holds position (zero actions). Action space = one fly
      (59-dim), drop-in for existing single-agent scripts.
  - opponent_mode='scripted': same action space; east_0 follows the
      ``goalie_policy`` callable and any fly in ``extra_policies`` follows
      its own callable. Anything without a policy holds still.
  - opponent_mode='self_play': the agent controls every fly. Action space
      = concat [west_0..west_{N-1}, east_0..east_{N-1}] (N*59 each side).

Teams:
  - ``n_per_team=1`` (default) is exactly the legacy setup: flies named
    ``attacker`` (west) and ``goalie`` (east), identical observation keys,
    so existing checkpoints keep loading.
  - ``n_per_team=N>1`` names flies ``west_0..west_{N-1}`` and
    ``east_0..east_{N-1}`` with an auto formation (or explicit
    ``west_spawns``/``east_spawns``) and team-colored thoraxes. MuJoCo has
    no text geoms, so jersey numbers live in the rollout video overlay
    and the GUI, keyed by these names.

Rewards:
  - ``get_reward`` is the scalar west-side reward (legacy behavior).
  - ``get_side_rewards`` returns ``{'west': r, 'east': r}`` with symmetric
    shaping for competitive/self-play training.
  - Game events (pass completed, shot on target, interception, save, goal)
    pay per-side bonuses and are exposed as a ``last_event`` one-hot
    observable plus a ``possession`` one-hot observable, so passing and
    shooting skills can emerge through training.
"""

import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_env import specs

from flybody.tasks.constants import (
    _WALK_PHYSICS_TIMESTEP,
    _WALK_CONTROL_TIMESTEP,
    _TERMINAL_QACC,
    _TERMINAL_LINVEL,
    _TERMINAL_ANGVEL,
)
from flybody.fruitfly.fruitfly import FruitFly
from flybody.utils import any_substr_in_str


_EVENT_NAMES = ("none", "pass", "shot", "interception", "save", "goal")
_TEAM_COLORS = {
    "west": (0.85, 0.18, 0.18, 1.0),
    "east": (0.18, 0.38, 0.95, 1.0),
}


class FootballFly(FruitFly):
    """FruitFly with a name-prefix-agnostic episode initializer.

    The stock ``FruitFly.initialize_episode`` hardcodes the ``'walker/'``
    body prefix, which breaks as soon as two flies (``attacker``/``goalie``)
    share one physics. This subclass does the same work (weight
    bookkeeping, wing retraction, prev_action reset) using the walker's own
    ``self.name`` prefix. It is used automatically for ``FruitFly``-based
    walkers; custom walker classes are left untouched.
    """

    def initialize_episode(self, physics, random_state):
        del random_state  # Unused.
        try:
            body_mass = physics.named.model.body_subtreemass[f"{self.name}/thorax"]
            self._weight = np.linalg.norm(physics.model.opt.gravity) * body_mass
        except KeyError:
            pass
        if not self._use_wings:
            for s in ["left", "right"]:
                for dof in ["yaw", "roll", "pitch"]:
                    j = f"{self.name}/wing_{dof}_{s}"
                    try:
                        physics.named.data.qpos[j] = physics.named.model.qpos_spring[j]
                    except KeyError:
                        pass
        self._prev_action = np.zeros_like(self._prev_action)


class FootballVs(composer.Task):
    """Two-team football task with configurable team size."""

    def __init__(
        self,
        walker,
        arena,
        time_limit: float = 10.0,
        force_actuators: bool = False,
        disable_wings: bool = True,
        joint_filter: float = 0.01,
        adhesion_filter: float = 0.007,
        opponent_mode: str = "static",
        goalie_policy=None,
        extra_policies: dict | None = None,
        n_per_team: int = 1,
        west_spawns: list | None = None,
        east_spawns: list | None = None,
        attacker_spawn=(-1.0, 0.0),
        goalie_spawn=None,
        ball_start=(0.0, 0.0),
        ball_spawn_noise: float = 0.15,
        spawn_noise: float = 0.1,
        goal_bonus: float = 10.0,
        concede_penalty: float = 10.0,
        pass_bonus: float = 1.0,
        shot_bonus: float = 0.5,
        interception_bonus: float = 1.0,
        save_bonus: float = 1.0,
        possession_radius: float = 0.35,
        kick_accel_thresh: float = 150.0,
        kick_radius: float = 0.45,
        shot_speed: float = 1.5,
        shot_cone: float = 0.45,
        pass_window: int = 40,
        possession_min_hold: int = 15,
        interception_cooldown: int = 50,
        event_hold: int = 25,
        team_colors: bool | None = None,
        extended_obs: bool = False,
        observables_options: dict | None = None,
    ):
        """Constructor.

        Args:
            walker: Walker constructor (flybody.fruitfly.fruitfly.FruitFly).
            arena: FootballArena instance.
            time_limit: Episode time limit in seconds.
            force_actuators: Whether to use force (vs position) actuators.
            disable_wings: Retract and disable wings on all flies.
            joint_filter: Timescale of joint actuator filter. 0: disabled.
            adhesion_filter: Timescale of adhesion actuator filter.
            opponent_mode: 'static', 'scripted', or 'self_play'.
            goalie_policy: Optional callable(step, physics) -> east_0 action
                used in scripted mode.
            extra_policies: Optional {fly_name: callable} for scripted mode.
                Any fly without a policy holds still.
            n_per_team: Flies per side. 1 keeps legacy attacker/goalie
                names; N>1 uses west_{i}/east_{i} names with an auto
                formation.
            west_spawns: Optional [(x, y)] * n_per_team overrides.
            east_spawns: Optional [(x, y)] * n_per_team overrides.
            attacker_spawn: (x, y) start for west_0 (legacy attacker).
            goalie_spawn: (x, y) start for east_0. Defaults to in front of
                east goal.
            ball_start: (x, y) nominal ball start.
            spawn_noise: Uniform noise (cm) added to fly spawns each episode.
            ball_spawn_noise: Uniform noise (cm) added to ball start x/y.
            goal_bonus: Sparse reward for scoring in the opponent goal.
            concede_penalty: Penalty (positive number, subtracted) for an
                own goal.
            pass_bonus: Bonus when a kick is collected by a teammate.
            shot_bonus: Bonus for a kick on target (fast, toward goal).
            interception_bonus: Bonus for taking possession from the other
                side after they held the ball.
            save_bonus: Bonus for redirecting a ball that was heading into
                your own goal.
            possession_radius: Nearest fly within this distance (cm) owns
                the ball.
            kick_accel_thresh: Ball acceleration (cm/s^2) above this counts
                as a kick by the nearest fly.
            kick_radius: Nearest fly must be within this distance (cm) to
                be credited with a kick.
            shot_speed: Minimum kick speed (cm/s) for a shot event.
            shot_cone: Max angle (rad) off the goal direction for a shot.
            pass_window: Control steps after a kick during which a teammate
                pickup counts as a completed pass.
            possession_min_hold: Steps the other side must hold the ball
                before a takeaway pays interception_bonus (anti-farming).
            interception_cooldown: Steps between interception bonuses.
            event_hold: Steps a game event stays visible in last_event.
            team_colors: Tint thoraxes red (west) / blue (east). Defaults
                to True when n_per_team > 1, False otherwise.
            extended_obs: Add ball_to_west_goal, possession, last_event and
                nearest-fly vectors. Auto-enabled when n_per_team > 1.
            observables_options: Passed to walker observables set_options.
        """
        if opponent_mode not in ("static", "scripted", "self_play"):
            raise ValueError(
                "opponent_mode must be 'static', 'scripted' or 'self_play'"
            )
        n_per_team = int(n_per_team)
        if n_per_team < 1:
            raise ValueError("n_per_team must be >= 1")
        self._n_per_team = n_per_team
        self._opponent_mode = opponent_mode
        self._goalie_policy = goalie_policy
        self._extra_policies = dict(extra_policies or {})
        self._time_limit = time_limit
        self._extended = bool(extended_obs) or n_per_team > 1

        if goalie_spawn is None:
            goalie_spawn = (arena.field_length / 2.0 - 0.5, 0.0)
        self._attacker_spawn = tuple(attacker_spawn)
        self._goalie_spawn = tuple(goalie_spawn)
        self._west_spawns = self._resolve_spawns(west_spawns, "west", n_per_team, arena)
        self._east_spawns = self._resolve_spawns(east_spawns, "east", n_per_team, arena)
        self._ball_start = tuple(ball_start)
        self._spawn_noise = spawn_noise
        self._ball_spawn_noise = ball_spawn_noise
        self._goal_bonus = goal_bonus
        self._concede_penalty = concede_penalty
        self._pass_bonus = pass_bonus
        self._shot_bonus = shot_bonus
        self._interception_bonus = interception_bonus
        self._save_bonus = save_bonus
        self._possession_radius = possession_radius
        self._kick_accel_thresh = kick_accel_thresh
        self._kick_radius = kick_radius
        self._shot_speed = shot_speed
        self._shot_cone = shot_cone
        self._pass_window = pass_window
        self._possession_min_hold = possession_min_hold
        self._interception_cooldown = interception_cooldown
        self._event_hold = event_hold

        self._arena = arena
        self._control_dt = _WALK_CONTROL_TIMESTEP
        self._step_counter = 0
        self._should_terminate = False
        self._scored = False
        self._conceded = False
        self._scored_east = False
        self._scored_west = False
        self.score = {"west": 0, "east": 0}
        self._init_goal_dist = 2.0
        self._init_goal_dist_side = {"west": 2.0, "east": 2.0}
        # Live game state (reset every episode).
        self._possession = None
        self._possession_fly = None
        self._poss_hold = 0
        self._prev_ball_vel = None
        self._last_kick = None  # (side, fly_name, step, vel)
        self._pass_credited_for = -(10**9)  # Kick step already credited.
        self._pending_bonus = {"west": 0.0, "east": 0.0}
        self._last_event = ("none", None, -(10**9))
        self._last_intercept_step = {"west": -(10**9), "east": -(10**9)}

        physics_timestep = _WALK_PHYSICS_TIMESTEP
        control_timestep = _WALK_CONTROL_TIMESTEP

        # --- Build the teams. ---
        # Stock FruitFly hardcodes the 'walker/' prefix in
        # initialize_episode; FootballFly fixes that for our team names.
        # Custom walker classes pass through.
        try:
            fly_cls = (
                FootballFly
                if isinstance(walker, type) and issubclass(walker, FruitFly)
                else walker
            )
        except TypeError:
            fly_cls = walker
        walker_kwargs = dict(
            use_legs=True,
            use_wings=not disable_wings,
            use_mouth=False,
            use_antennae=False,
            force_actuators=force_actuators,
            joint_filter=joint_filter,
            adhesion_filter=adhesion_filter,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )
        if n_per_team == 1:
            west_names, east_names = ["attacker"], ["goalie"]
        else:
            west_names = [f"west_{i}" for i in range(n_per_team)]
            east_names = [f"east_{i}" for i in range(n_per_team)]
        self._west = [fly_cls(name=n, **walker_kwargs) for n in west_names]
        self._east = [fly_cls(name=n, **walker_kwargs) for n in east_names]
        self._flies = {"west": self._west, "east": self._east}
        self._fly_order = self._west + self._east
        self._fly_by_name = {w.name: w for w in self._fly_order}
        # Legacy compat: attacker = west_0, goalie = east_0.
        self._attacker = self._west[0]
        self._goalie = self._east[0]
        for w in self._fly_order:
            if observables_options is not None:
                w.observables.set_options(observables_options)

        # Team colors: tint thorax geoms red/blue (visual only, no physics
        # change). MuJoCo has no text geoms; jersey numbers live in the
        # rollout video overlay and the GUI, keyed by fly name.
        if team_colors is None:
            team_colors = n_per_team > 1
        if team_colors:
            self._paint_teams()

        # Attach all flies to the arena at their spawn sites. West faces
        # +x (east goal), east faces -x.
        self._spawns = {}
        for side, flies, spawns in (
            ("west", self._west, self._west_spawns),
            ("east", self._east, self._east_spawns),
        ):
            for w, spawn in zip(flies, spawns):
                self._spawns[w.name] = tuple(spawn)
                spawn_pos = np.array([spawn[0], spawn[1], w.upright_pose.xpos[2]])
                spawn_site = self._arena.mjcf_model.worldbody.add("site", pos=spawn_pos)
                w.create_root_joints(self._arena.attach(w, spawn_site))
                spawn_site.remove()

        self._root_joints = {
            w.name: mjcf.get_frame_freejoint(w.mjcf_model) for w in self._fly_order
        }

        # Floor contact params (same as Walking base).
        for geom in self._arena.ground_geoms:
            geom.friction = (0.5,)
            geom.solref = (0.001, 1)
            geom.solimp = (0.95, 0.99, 0.01)

        # Exclude wing-leg collisions per walker.
        for w in self._fly_order:
            contact = w.mjcf_model.contact
            for body in w.mjcf_model.find_all("body"):
                if any_substr_in_str(
                    ["coxa", "femur", "tibia", "tarsus", "claw"], body.name
                ):
                    for wing in ["wing_left", "wing_right"]:
                        contact.add(
                            "exclude",
                            name=f"{w.name}_{body.name}_{wing}",
                            body1=body.name,
                            body2=wing,
                        )

        # Retracted-wing springrefs (for reward + init).
        self._wing_joints = {}
        self._wing_springrefs = {}
        for w in self._fly_order:
            joints, springrefs = [], []
            for joint in w.mjcf_model.find_all("joint"):
                if any_substr_in_str(["yaw", "roll", "pitch"], joint.name):
                    springref = joint.springref or joint.dclass.joint.springref or 0.0
                    joints.append(joint)
                    springrefs.append(springref)
            self._wing_joints[w.name] = joints
            self._wing_springrefs[w.name] = np.asarray(springrefs, dtype=float)

        # Shared game-state observables (visible under attacker/... =
        # west_0, exactly as before for n_per_team=1).
        for obs_name in (
            "ball_pos",
            "ball_vel",
            "attacker_to_ball",
            "ball_to_east_goal",
            "goalie_to_ball",
        ):
            self._attacker.observables.add_observable(obs_name, getattr(self, obs_name))
        if self._extended:
            for obs_name in (
                "ball_to_west_goal",
                "possession",
                "last_event",
                "nearest_west_to_ball",
                "nearest_east_to_ball",
            ):
                self._attacker.observables.add_observable(
                    obs_name, getattr(self, obs_name)
                )

        # Enable standard walking observables on all flies.
        for w in self._fly_order:
            for sensor in w.observables.vestibular + w.observables.proprioception:
                sensor.enabled = True
            w.observables.appendages_pos.enabled = True
            w.observables.force.enabled = True
            w.observables.touch.enabled = True
            try:
                w.observables.self_contact.enabled = False
            except Exception:  # pylint: disable=broad-except
                pass

        # Correct fly mass bounds.
        for w in self._fly_order:
            w.mjcf_model.compiler.boundmass = 0.0
            w.mjcf_model.compiler.boundinertia = 0.0

        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep
        )

    def _resolve_spawns(self, spawns, side, n, arena):
        if spawns is not None:
            spawns = [tuple(s) for s in spawns]
            if len(spawns) != n:
                raise ValueError(f"{side}_spawns must have n_per_team={n} entries")
            return spawns
        if n == 1:
            if side == "west":
                return [self._attacker_spawn]
            return [tuple(self._goalie_spawn)]
        # Auto formation: index 0 holds the legacy goal line (east) or
        # striker spot (west); the rest fan out behind / across.
        out = []
        for i in range(n):
            y = (i - (n - 1) / 2.0) * 0.9
            if side == "west":
                out.append((-1.0 - i * 0.7, y))
            else:
                out.append((arena.field_length / 2.0 - 0.5 - i * 0.7, y))
        return out

    def _paint_teams(self):
        """Tint each fly's thorax with its team color (visual only).

        The material lives in the walker's own asset namespace under a
        per-fly name, so team paint never leaks across flies and needs no
        cross-model references. MuJoCo has no text geoms; jersey numbers
        live in the rollout video overlay and the GUI, keyed by fly name.
        """
        for side, flies in self._flies.items():
            for w in flies:
                mat_name = f"{w.name}_jersey"
                try:
                    w.mjcf_model.asset.add(
                        "material", name=mat_name, rgba=list(_TEAM_COLORS[side])
                    )
                except Exception:  # pylint: disable=broad-except
                    pass
                try:
                    thorax = w.mjcf_model.find("body", "thorax")
                    geoms = thorax.find_all("geom")
                except Exception:  # pylint: disable=broad-except
                    continue
                for g in geoms:
                    try:
                        g.material = mat_name
                    except Exception:  # pylint: disable=broad-except
                        pass

    # -- Composer API. --
    @property
    def root_entity(self):
        return self._arena

    @property
    def walker(self):
        # Main (west_0) walker for compat with training scripts.
        return self._attacker

    @property
    def attacker(self):
        return self._attacker

    @property
    def goalie(self):
        return self._goalie

    @property
    def west(self):
        return list(self._west)

    @property
    def east(self):
        return list(self._east)

    @property
    def n_per_team(self):
        return self._n_per_team

    @property
    def possession_side(self):
        """'west', 'east' or None per the last game-state update."""
        return self._possession

    @property
    def last_event_info(self):
        """(event_name, side, step) of the latest game event."""
        return self._last_event

    def initialize_episode_mjcf(self, random_state):
        if hasattr(self._arena, "regenerate"):
            self._arena.regenerate(random_state)
        self.root_entity.mjcf_model.visual.map.znear = 0.001
        self.root_entity.mjcf_model.visual.map.zfar = 50.0
        self.root_entity.mjcf_model.visual.map.force = 0.00001
        self.root_entity.mjcf_model.statistic.extent = 4.01

    def initialize_episode(self, physics, random_state):
        self._step_counter = 0
        self._should_terminate = False
        self._scored = False
        self._conceded = False
        self._scored_east = False
        self._scored_west = False
        self.score = {"west": 0, "east": 0}
        self._possession = None
        self._possession_fly = None
        self._poss_hold = 0
        self._prev_ball_vel = None
        self._last_kick = None
        self._pass_credited_for = -(10**9)
        self._pending_bonus = {"west": 0.0, "east": 0.0}
        self._last_event = ("none", None, -(10**9))
        self._last_intercept_step = {"west": -(10**9), "east": -(10**9)}

        spawn_z = float(self._attacker.upright_pose.xpos[2])
        # Lateral spawn noise forces the policy to cope with varied
        # starting geometries instead of memorizing one run-up.
        for w in self._fly_order:
            side = "west" if w in self._west else "east"
            sx, sy = self._spawns[w.name]
            x = sx + random_state.uniform(-self._spawn_noise, self._spawn_noise)
            y = sy + random_state.uniform(-self._spawn_noise, self._spawn_noise)
            if side == "west":
                quat = [1, 0, 0, 0]  # Face +x (east goal).
            else:
                quat = [0, 0, 0, 1]  # 180 deg yaw, face -x.
            physics.bind(self._root_joints[w.name]).qpos = np.array(
                [x, y, spawn_z] + quat
            )
            physics.bind(self._root_joints[w.name]).qvel = np.zeros(6)

        # Ball at nominal start + noise, resting on floor.
        bx = self._ball_start[0] + random_state.uniform(
            -self._ball_spawn_noise, self._ball_spawn_noise
        )
        by = self._ball_start[1] + random_state.uniform(
            -self._ball_spawn_noise, self._ball_spawn_noise
        )
        ball_qpos = np.array([bx, by, self._arena.ball_radius + 0.01, 1, 0, 0, 0])
        physics.named.data.qpos["football"] = ball_qpos
        physics.named.data.qvel["football"] = np.zeros(6)

        # Reference distances for ball-progress rewards (per side).
        east_goal = np.array([self._arena.east_goal_x, 0.0, self._arena.ball_radius])
        west_goal = np.array([self._arena.west_goal_x, 0.0, self._arena.ball_radius])
        self._init_goal_dist = float(
            np.linalg.norm(np.asarray(ball_qpos[:3]) - east_goal)
        )
        self._init_goal_dist_side = {
            "west": float(np.linalg.norm(np.asarray(ball_qpos[:3]) - east_goal)),
            "east": float(np.linalg.norm(np.asarray(ball_qpos[:3]) - west_goal)),
        }

        # Retract wings (bind MJCF elements directly, as in WalkImitation).
        # FootballFly entity hooks repeat this themselves; kept here so the
        # pose is correct regardless of hook ordering.
        for w in self._fly_order:
            if self._wing_joints[w.name]:
                physics.bind(self._wing_joints[w.name]).qpos = self._wing_springrefs[
                    w.name
                ]

    def _split_action(self, action):
        action = np.asarray(action, dtype=float)
        half = len(action) // 2
        return action[:half], action[half:]

    def before_step(self, physics, action, random_state):
        self._step_counter += 1
        # Fresh event bonuses every control step; get_*_reward only reads.
        self._pending_bonus = {"west": 0.0, "east": 0.0}
        self._detect_events(physics)

        if self._opponent_mode == "self_play":
            dims = len(action) // (2 * self._n_per_team)
            chunks = [
                action[i * dims : (i + 1) * dims] for i in range(2 * self._n_per_team)
            ]
            for w, a in zip(self._fly_order, chunks):
                self._apply_action_by_name(physics, w, a)
        else:
            self._apply_action_by_name(physics, self._attacker, action)
            for w in self._fly_order:
                if w is self._attacker:
                    continue
                policy = self._extra_policies.get(w.name)
                if (
                    policy is None
                    and self._opponent_mode == "scripted"
                    and w is self._goalie
                    and self._goalie_policy is not None
                ):
                    policy = self._goalie_policy
                if policy is not None:
                    w_action = np.asarray(
                        policy(self._step_counter, physics), dtype=float
                    )
                    self._apply_action_by_name(physics, w, w_action)
                else:
                    # Hold pose (zeros = no delta for position actuators).
                    n_act = self._fly_action_dim(physics, w)
                    self._apply_action_by_name(physics, w, np.zeros(n_act))

    def _fly_action_dim(self, physics, walker):
        try:
            return walker.get_action_spec(physics).shape[0]
        except Exception:  # pylint: disable=broad-except
            return self._attacker.get_action_spec(physics).shape[0]

    def _apply_action_by_name(self, physics, walker, action):
        """Apply a walker's action without wiping the other walkers' ctrl.

        All flies share one global ctrl vector. The stock
        ``walker.apply_action`` builds a fresh zero vector from the walker's
        *local* actuator indices, which would erase the other flies'
        controls and mis-address their actuators. Here we map each action
        entry to its global ``<walker>/<actuator>`` ctrl slot by name.
        """
        action = np.asarray(action, dtype=float)
        local_names = [a.name for a in walker.mjcf_model.find_all("actuator")]
        for key in walker._action_indices.keys():
            if key == "user":
                continue  # No MuJoCo actuator behind user actions.
            a_idx = walker._action_indices[key]
            c_idx = walker._ctrl_indices[key]
            if not c_idx or not a_idx:
                continue
            for ap, cl in zip(a_idx, c_idx):
                global_name = f"{walker.name}/{local_names[cl]}"
                try:
                    physics.named.data.ctrl[global_name] = action[ap]
                except KeyError:
                    pass
        walker._prev_action[:] = action

    def action_spec(self, physics):
        a_spec = self._attacker.get_action_spec(physics)
        if self._opponent_mode != "self_play":
            return a_spec
        parts_min, parts_max = [a_spec.minimum], [a_spec.maximum]
        for w in self._fly_order[1:]:
            w_spec = w.get_action_spec(physics)
            parts_min.append(w_spec.minimum)
            parts_max.append(w_spec.maximum)
        minimum = np.concatenate(parts_min)
        maximum = np.concatenate(parts_max)
        return specs.BoundedArray(
            shape=(minimum.shape[0],),
            dtype=float,
            minimum=minimum,
            maximum=maximum,
            name="west+east",
        )

    # -- Game logic helpers. --
    def _ball_state(self, physics):
        ball_qpos = np.asarray(physics.named.data.qpos["football"])
        ball_qvel = np.asarray(physics.named.data.qvel["football"])
        return ball_qpos[:3], ball_qvel[:3]

    def _walker_pos(self, physics, walker):
        pos, _ = walker.get_pose(physics)
        return np.asarray(pos)

    def _nearest_fly(self, physics, pos):
        """(name, side, distance) of the fly closest to a point."""
        best = (None, None, np.inf)
        for side, flies in self._flies.items():
            for w in flies:
                d = float(
                    np.linalg.norm(self._walker_pos(physics, w) - np.asarray(pos))
                )
                if d < best[2]:
                    best = (w.name, side, d)
        return best

    def _opp_goal(self, side):
        if side == "west":
            return np.array([self._arena.east_goal_x, 0.0, self._arena.ball_radius])
        return np.array([self._arena.west_goal_x, 0.0, self._arena.ball_radius])

    def _own_goal_sign(self, side):
        return -1.0 if side == "west" else 1.0

    def _in_goal(self, ball_pos, side="east"):
        gx = self._arena.east_goal_x if side == "east" else self._arena.west_goal_x
        crossed = (ball_pos[0] > gx) if side == "east" else (ball_pos[0] < gx)
        return bool(
            crossed
            and abs(ball_pos[1]) < self._arena.goal_width / 2.0
            and ball_pos[2] < self._arena.goal_height
        )

    def _record_event(self, name, side):
        self._last_event = (name, side, self._step_counter)

    def _credit_pass(self, side, fly_name):
        """Credit a completed pass once per kick (no farming)."""
        if self._last_kick is None:
            return
        kside, kicker, kstep, _ = self._last_kick
        if (
            kside == side
            and fly_name != kicker
            and self._step_counter - kstep <= self._pass_window
            and self._pass_credited_for != kstep
        ):
            self._pending_bonus[side] += self._pass_bonus
            self._pass_credited_for = kstep
            self._record_event("pass", side)

    def _detect_events(self, physics):
        """Kick/pass/shot/interception/save detection, once per step."""
        ball_pos, ball_vel = self._ball_state(physics)
        prev_vel = np.zeros(3) if self._prev_ball_vel is None else self._prev_ball_vel
        accel = float(np.linalg.norm(ball_vel - prev_vel)) / self._control_dt

        # --- Kick: sudden ball acceleration next to a fly. ---
        if accel > self._kick_accel_thresh:
            kicker, kside, kdist = self._nearest_fly(physics, ball_pos)
            if kicker is not None and kdist < self._kick_radius:
                speed = float(np.linalg.norm(ball_vel))
                self._last_kick = (
                    kside,
                    kicker,
                    self._step_counter,
                    np.asarray(ball_vel).copy(),
                )
                # Shot on target: fast kick toward the opponent goal.
                to_goal = self._opp_goal(kside) - ball_pos
                cos_ang = float(
                    np.dot(ball_vel, to_goal)
                    / (
                        (np.linalg.norm(ball_vel) + 1e-8)
                        * (np.linalg.norm(to_goal) + 1e-8)
                    )
                )
                if speed > self._shot_speed and cos_ang > np.cos(self._shot_cone):
                    self._pending_bonus[kside] += self._shot_bonus
                    self._record_event("shot", kside)
                # Save: redirecting a ball that was heading into your own
                # goal while it is in your defensive third.
                third = self._arena.field_length / 6.0
                in_own_third = (
                    ball_pos[0] < -third if kside == "west" else ball_pos[0] > third
                )
                was_own_way = (
                    (prev_vel[0] < -0.5) if kside == "west" else (prev_vel[0] > 0.5)
                )
                if in_own_third and was_own_way:
                    self._pending_bonus[kside] += self._save_bonus
                    self._record_event("save", kside)

        # --- Possession: nearest fly inside the radius owns the ball. ---
        pname, pside, pdist = self._nearest_fly(physics, ball_pos)
        poss = pside if pdist < self._possession_radius else None
        poss_fly = pname if poss is not None else None
        prev_poss = self._possession
        if poss != prev_poss:
            if poss is not None and prev_poss is not None and poss != prev_poss:
                # Turnover between sides: interception if the other side
                # held the ball (anti-farming) and cooldown elapsed.
                if (
                    self._poss_hold >= self._possession_min_hold
                    and (self._step_counter - self._last_intercept_step[poss])
                    > self._interception_cooldown
                ):
                    self._pending_bonus[poss] += self._interception_bonus
                    self._last_intercept_step[poss] = self._step_counter
                    self._record_event("interception", poss)
            elif poss is not None:
                self._credit_pass(poss, poss_fly)
            self._possession, self._possession_fly = poss, poss_fly
            self._poss_hold = 0
        elif poss is not None:
            self._poss_hold += 1
            self._credit_pass(poss, poss_fly)
        else:
            self._poss_hold = 0

        # Copy: named qvel access aliases the live physics buffer, which
        # mutates in place every step (a stored view would always equal
        # the current velocity and no kick would ever be detected).
        self._prev_ball_vel = np.asarray(ball_vel).copy()

    def _side_dense(self, physics, side):
        """Five shaped factors for one side (mirrors legacy west math)."""
        own = self._flies[side]
        ball_pos, ball_vel = self._ball_state(physics)
        goal = self._opp_goal(side)
        goal_dir_sign = 1.0 if side == "west" else -1.0

        # 1. Chase: nearest own fly to the ball.
        d_near, near_w = np.inf, own[0]
        for w in own:
            d = float(np.linalg.norm(self._walker_pos(physics, w) - ball_pos))
            if d < d_near:
                d_near, near_w = d, w
        approach = float(np.exp(-3.0 * d_near))

        # 2. Ball closeness to the opponent goal.
        d_ball_goal = float(np.linalg.norm(ball_pos - goal))
        ball_to_goal = float(np.exp(-1.5 * d_ball_goal))

        # 3. Ball progress vs episode start (positive only when closer).
        progress = float(np.tanh(self._init_goal_dist_side[side] - d_ball_goal))

        # 4. Signed ball velocity toward the opponent goal.
        kick = float(np.tanh(3.0 * ball_vel[0] * goal_dir_sign))

        # 5. Nearest own fly's root velocity toward the ball.
        root_vel = np.asarray(physics.bind(self._root_joints[near_w.name]).qvel[:3])
        to_ball = ball_pos - self._walker_pos(physics, near_w)
        dist = np.linalg.norm(to_ball) + 1e-8
        chase_vel = float(np.tanh(2.0 * np.dot(root_vel, to_ball / dist)))

        return np.array([approach, ball_to_goal, progress, kick, chase_vel])

    def get_reward_factors(self, physics):
        """Shaped factors. Standing still scores ~0; moving the ball
        toward the east goal is the only way to earn."""
        factors = self._side_dense(physics, "west")
        if self._scored:
            factors = np.append(factors, self._goal_bonus)
        if self._conceded:
            factors = np.append(factors, -self._concede_penalty)
        return factors

    def get_reward(self, physics):
        factors = self.get_reward_factors(physics)
        dense = float(
            1.0 * factors[0]
            + 1.0 * factors[1]
            + 1.0 * factors[2]
            + 1.0 * factors[3]
            + 0.5 * factors[4]
        )
        sparse = float(np.sum(factors[5:])) if len(factors) > 5 else 0.0
        sparse += float(self._pending_bonus["west"])
        self._should_terminate = self.check_termination(physics)
        return dense + sparse

    def get_side_rewards(self, physics):
        """Per-team total reward for competitive/self-play training."""
        out = {}
        for side in ("west", "east"):
            dense_factors = self._side_dense(physics, side)
            dense = float(
                1.0 * dense_factors[0]
                + 1.0 * dense_factors[1]
                + 1.0 * dense_factors[2]
                + 1.0 * dense_factors[3]
                + 0.5 * dense_factors[4]
            )
            sparse = 0.0
            if side == "west":
                if self._scored_east:
                    sparse += self._goal_bonus
                if self._scored_west:
                    sparse -= self._concede_penalty
            else:
                if self._scored_west:
                    sparse += self._goal_bonus
                if self._scored_east:
                    sparse -= self._concede_penalty
            sparse += float(self._pending_bonus[side])
            out[side] = dense + sparse
        self._should_terminate = self.check_termination(physics)
        return out

    def check_termination(self, physics):
        ball_pos, _ = self._ball_state(physics)
        if self._in_goal(ball_pos, "east"):
            self._scored = True
            self._scored_east = True
            self.score["west"] += 1
            self._record_event("goal", "west")
            return True
        if self._in_goal(ball_pos, "west"):
            self._conceded = True
            self._scored_west = True
            self.score["east"] += 1
            self._record_event("goal", "east")
            return True
        # Out of bounds (missed everything).
        if (
            abs(ball_pos[0]) > self._arena.field_length / 2.0 + 1.0
            or abs(ball_pos[1]) > self._arena.field_width / 2.0 + 1.0
        ):
            return True
        # Flip / explosion guards (checked on west_0, as before).
        try:
            att_vel = np.linalg.norm(self._attacker.observables.velocimeter(physics))
            att_ang = np.linalg.norm(self._attacker.observables.gyro(physics))
            if att_vel > _TERMINAL_LINVEL or att_ang > _TERMINAL_ANGVEL:
                return True
        except Exception:  # pylint: disable=broad-except
            pass
        if np.linalg.norm(np.asarray(physics.data.qacc)) > _TERMINAL_QACC:
            return True
        return False

    def should_terminate_episode(self, physics):
        return self._should_terminate

    def get_discount(self, physics):
        del physics
        return 0.0 if self._should_terminate else 1.0

    # -- Task observables (shared game state). --
    @composer.observable
    def ball_pos(self):
        def get_ball_pos(physics):
            pos, _ = self._ball_state(physics)
            return np.asarray(pos)

        return observable.Generic(get_ball_pos)

    @composer.observable
    def ball_vel(self):
        def get_ball_vel(physics):
            _, vel = self._ball_state(physics)
            return np.asarray(vel)

        return observable.Generic(get_ball_vel)

    @composer.observable
    def attacker_to_ball(self):
        def get_vec(physics):
            return np.asarray(
                physics.named.data.qpos["football"][:3]
            ) - self._walker_pos(physics, self._attacker)

        return observable.Generic(get_vec)

    @composer.observable
    def ball_to_east_goal(self):
        def get_vec(physics):
            ball = np.asarray(physics.named.data.qpos["football"][:3])
            return (
                np.array(
                    [
                        self._arena.east_goal_x,
                        0.0,
                        self._arena.ball_radius,
                    ]
                )
                - ball
            )

        return observable.Generic(get_vec)

    @composer.observable
    def goalie_to_ball(self):
        def get_vec(physics):
            return np.asarray(
                physics.named.data.qpos["football"][:3]
            ) - self._walker_pos(physics, self._goalie)

        return observable.Generic(get_vec)

    @composer.observable
    def ball_to_west_goal(self):
        def get_vec(physics):
            ball = np.asarray(physics.named.data.qpos["football"][:3])
            return (
                np.array(
                    [
                        self._arena.west_goal_x,
                        0.0,
                        self._arena.ball_radius,
                    ]
                )
                - ball
            )

        return observable.Generic(get_vec)

    @composer.observable
    def possession(self):
        """One-hot [west, east, none] from the current physics state."""

        def get_poss(physics):
            ball_pos, _ = self._ball_state(physics)
            _, side, dist = self._nearest_fly(physics, ball_pos)
            onehot = np.zeros(3)
            if dist < self._possession_radius:
                onehot[0 if side == "west" else 1] = 1.0
            else:
                onehot[2] = 1.0
            return onehot

        return observable.Generic(get_poss)

    @composer.observable
    def last_event(self):
        """One-hot over (none, pass, shot, interception, save, goal)."""

        def get_event(physics):
            del physics
            onehot = np.zeros(len(_EVENT_NAMES))
            name, _, step = self._last_event
            if (
                name in _EVENT_NAMES[1:]
                and self._step_counter - step <= self._event_hold
            ):
                onehot[_EVENT_NAMES.index(name)] = 1.0
            else:
                onehot[0] = 1.0
            return onehot

        return observable.Generic(get_event)

    @composer.observable
    def nearest_west_to_ball(self):
        def get_vec(physics):
            ball = np.asarray(physics.named.data.qpos["football"][:3])
            best, best_w = np.inf, self._west[0]
            for w in self._west:
                d = float(np.linalg.norm(self._walker_pos(physics, w) - ball))
                if d < best:
                    best, best_w = d, w
            return ball - self._walker_pos(physics, best_w)

        return observable.Generic(get_vec)

    @composer.observable
    def nearest_east_to_ball(self):
        def get_vec(physics):
            ball = np.asarray(physics.named.data.qpos["football"][:3])
            best, best_w = np.inf, self._east[0]
            for w in self._east:
                d = float(np.linalg.norm(self._walker_pos(physics, w) - ball))
                if d < best:
                    best, best_w = d, w
            return ball - self._walker_pos(physics, best_w)

        return observable.Generic(get_vec)
