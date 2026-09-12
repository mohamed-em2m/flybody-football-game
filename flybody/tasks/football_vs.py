"""Fly-vs-fly football (soccer) goal-scoring task.

Attacker fly tries to push the ball into the EAST goal (+x).
Goalie/defender fly tries to stop it and can score in the WEST goal (-x).
Both flies live in the same environment (same physics).

Modes:
  - opponent_mode='static': goalie holds position (zero actions).
      Action space = attacker only (59-dim), drop-in for existing DMPO script.
  - opponent_mode='scripted': same as static v1 (goalie holds + optional
      user goalie_policy callable). Action space = attacker only.
  - opponent_mode='self_play': both flies controlled by the agent.
      Action space = concat [attacker, goalie] (118-dim) for self-play /
      competitive training.
"""

import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.utils import rewards
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
            body_mass = physics.named.model.body_subtreemass[
                f'{self.name}/thorax']
            self._weight = np.linalg.norm(
                physics.model.opt.gravity) * body_mass
        except KeyError:
            pass
        if not self._use_wings:
            for s in ['left', 'right']:
                for dof in ['yaw', 'roll', 'pitch']:
                    j = f'{self.name}/wing_{dof}_{s}'
                    try:
                        physics.named.data.qpos[j] = \
                            physics.named.model.qpos_spring[j]
                    except KeyError:
                        pass
        self._prev_action = np.zeros_like(self._prev_action)


class FootballVs(composer.Task):
    """Two-fly football task."""

    def __init__(self,
                 walker,
                 arena,
                 time_limit: float = 10.0,
                 force_actuators: bool = False,
                 disable_wings: bool = True,
                 joint_filter: float = 0.01,
                 adhesion_filter: float = 0.007,
                 opponent_mode: str = 'static',
                 goalie_policy=None,
                 attacker_spawn=(-1.0, 0.0),
                 goalie_spawn=None,
                 ball_spawn_noise: float = 0.15,
                 goal_bonus: float = 10.0,
                 concede_penalty: float = 10.0,
                 observables_options: dict | None = None):
        """Constructor.

        Args:
            walker: Walker constructor (flybody.fruitfly.fruitfly.FruitFly).
            arena: FootballArena instance.
            time_limit: Episode time limit in seconds.
            force_actuators: Whether to use force (vs position) actuators.
            disable_wings: Retract and disable wings on both flies.
            joint_filter: Timescale of joint actuator filter. 0: disabled.
            adhesion_filter: Timescale of adhesion actuator filter.
            opponent_mode: 'static', 'scripted', or 'self_play'.
            goalie_policy: Optional callable(step, physics) -> goalie action
                used in scripted mode. If None, goalie holds (zeros).
            attacker_spawn: (x, y) start for attacker.
            goalie_spawn: (x, y) start for goalie. Defaults to in front of
                east goal.
            ball_spawn_noise: Uniform noise (cm) added to ball start x/y.
            goal_bonus: Sparse reward for scoring in east goal.
            concede_penalty: Penalty (positive number, subtracted) when ball
                enters west goal (own goal / goalie scores).
            observables_options: Passed to walker observables set_options.
        """
        if opponent_mode not in ('static', 'scripted', 'self_play'):
            raise ValueError(
                "opponent_mode must be 'static', 'scripted' or 'self_play'")
        self._opponent_mode = opponent_mode
        self._goalie_policy = goalie_policy
        self._time_limit = time_limit
        self._attacker_spawn = tuple(attacker_spawn)
        if goalie_spawn is None:
            goalie_spawn = (arena.field_length / 2.0 - 0.5, 0.0)
        self._goalie_spawn = tuple(goalie_spawn)
        self._ball_spawn_noise = ball_spawn_noise
        self._goal_bonus = goal_bonus
        self._concede_penalty = concede_penalty

        self._arena = arena
        self._step_counter = 0
        self._should_terminate = False
        self._scored = False
        self._conceded = False

        physics_timestep = _WALK_PHYSICS_TIMESTEP
        control_timestep = _WALK_CONTROL_TIMESTEP

        # --- Build two walkers. ---
        # Stock FruitFly hardcodes the 'walker/' prefix in
        # initialize_episode; FootballFly fixes that for our
        # attacker/goalie names. Custom walker classes pass through.
        try:
            fly_cls = (FootballFly if isinstance(walker, type)
                       and issubclass(walker, FruitFly) else walker)
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
        self._attacker = fly_cls(name='attacker', **walker_kwargs)
        self._goalie = fly_cls(name='goalie', **walker_kwargs)
        for w in (self._attacker, self._goalie):
            if observables_options is not None:
                w.observables.set_options(observables_options)

        # Attach both to arena at their spawn sites.
        for w, spawn in ((self._attacker, self._attacker_spawn),
                         (self._goalie, self._goalie_spawn)):
            spawn_pos = np.array(
                [spawn[0], spawn[1], w.upright_pose.xpos[2]])
            spawn_site = self._arena.mjcf_model.worldbody.add(
                'site', pos=spawn_pos)
            w.create_root_joints(self._arena.attach(w, spawn_site))
            spawn_site.remove()

        self._root_joints = {
            'attacker': mjcf.get_frame_freejoint(
                self._attacker.mjcf_model),
            'goalie': mjcf.get_frame_freejoint(self._goalie.mjcf_model),
        }

        # Floor contact params (same as Walking base).
        for geom in self._arena.ground_geoms:
            geom.friction = (0.5,)
            geom.solref = (0.001, 1)
            geom.solimp = (0.95, 0.99, 0.01)

        # Exclude wing-leg collisions per walker.
        for w in (self._attacker, self._goalie):
            contact = w.mjcf_model.contact
            for body in w.mjcf_model.find_all('body'):
                if any_substr_in_str(
                    ['coxa', 'femur', 'tibia', 'tarsus', 'claw'],
                        body.name):
                    for wing in ['wing_left', 'wing_right']:
                        contact.add('exclude',
                                    name=f'{w.name}_{body.name}_{wing}',
                                    body1=body.name,
                                    body2=wing)

        # Retracted-wing springrefs (for reward + init).
        self._wing_joints = {}
        self._wing_springrefs = {}
        for key, w in (('attacker', self._attacker),
                       ('goalie', self._goalie)):
            joints, springrefs = [], []
            for joint in w.mjcf_model.find_all('joint'):
                if any_substr_in_str(['yaw', 'roll', 'pitch'], joint.name):
                    springref = (joint.springref
                                 or joint.dclass.joint.springref or 0.)
                    joints.append(joint)
                    springrefs.append(springref)
            self._wing_joints[key] = joints
            self._wing_springrefs[key] = np.asarray(springrefs,
                                                    dtype=float)

        # Shared game-state observables (visible under attacker/...).
        for obs_name in ('ball_pos', 'ball_vel', 'attacker_to_ball',
                         'ball_to_east_goal', 'goalie_to_ball'):
            self._attacker.observables.add_observable(
                obs_name, getattr(self, obs_name))

        # Enable standard walking observables on both flies.
        for w in (self._attacker, self._goalie):
            for sensor in (w.observables.vestibular +
                           w.observables.proprioception):
                sensor.enabled = True
            w.observables.appendages_pos.enabled = True
            w.observables.force.enabled = True
            w.observables.touch.enabled = True
            try:
                w.observables.self_contact.enabled = False
            except Exception:  # pylint: disable=broad-except
                pass

        # Correct fly mass bounds.
        for w in (self._attacker, self._goalie):
            w.mjcf_model.compiler.boundmass = 0.
            w.mjcf_model.compiler.boundinertia = 0.

        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

    # -- Composer API. --
    @property
    def root_entity(self):
        return self._arena

    @property
    def walker(self):
        # Main (attacker) walker for compat with training scripts.
        return self._attacker

    @property
    def attacker(self):
        return self._attacker

    @property
    def goalie(self):
        return self._goalie

    def initialize_episode_mjcf(self, random_state):
        if hasattr(self._arena, 'regenerate'):
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

        spawn_z = float(self._attacker.upright_pose.xpos[2])
        # Attacker faces +x (east goal), goalie faces -x.
        attacker_qpos = np.array([
            self._attacker_spawn[0], self._attacker_spawn[1], spawn_z,
            1, 0, 0, 0,
        ])
        goalie_qpos = np.array([
            self._goalie_spawn[0], self._goalie_spawn[1], spawn_z,
            0, 0, 0, 1,  # 180 deg yaw.
        ])
        physics.bind(self._root_joints['attacker']).qpos = attacker_qpos
        physics.bind(self._root_joints['goalie']).qpos = goalie_qpos
        physics.bind(self._root_joints['attacker']).qvel = np.zeros(6)
        physics.bind(self._root_joints['goalie']).qvel = np.zeros(6)

        # Ball at center + noise, resting on floor.
        bx = random_state.uniform(-self._ball_spawn_noise,
                                  self._ball_spawn_noise)
        by = random_state.uniform(-self._ball_spawn_noise,
                                  self._ball_spawn_noise)
        ball_qpos = np.array(
            [bx, by, self._arena.ball_radius + 0.01, 1, 0, 0, 0])
        physics.named.data.qpos['football'] = ball_qpos
        physics.named.data.qvel['football'] = np.zeros(6)

        # Retract wings (bind MJCF elements directly, as in WalkImitation).
        # FootballFly entity hooks repeat this themselves; kept here so the
        # pose is correct regardless of hook ordering.
        if self._wing_joints['attacker']:
            physics.bind(self._wing_joints['attacker']).qpos = \
                self._wing_springrefs['attacker']
        if self._wing_joints['goalie']:
            physics.bind(self._wing_joints['goalie']).qpos = \
                self._wing_springrefs['goalie']

    def _split_action(self, action):
        action = np.asarray(action, dtype=float)
        half = len(action) // 2
        return action[:half], action[half:]

    def before_step(self, physics, action, random_state):
        self._step_counter += 1
        if self._opponent_mode == 'self_play':
            attacker_action, goalie_action = self._split_action(action)
            self._apply_action_by_name(physics, self._attacker,
                                       attacker_action)
            self._apply_action_by_name(physics, self._goalie, goalie_action)
        else:
            self._apply_action_by_name(physics, self._attacker, action)
            if (self._opponent_mode == 'scripted'
                    and self._goalie_policy is not None):
                goalie_action = np.asarray(
                    self._goalie_policy(self._step_counter, physics),
                    dtype=float)
                self._apply_action_by_name(physics, self._goalie,
                                           goalie_action)
            else:
                # static: hold pose (zeros = no delta for position actuators).
                n_g = self._goalie_action_dim(physics)
                self._apply_action_by_name(physics, self._goalie,
                                           np.zeros(n_g))

    def _goalie_action_dim(self, physics):
        try:
            return self._goalie.get_action_spec(physics).shape[0]
        except Exception:  # pylint: disable=broad-except
            return self._attacker.get_action_spec(physics).shape[0]

    def _apply_action_by_name(self, physics, walker, action):
        """Apply a walker's action without wiping the other walker's ctrl.

        Both flies share one global ctrl vector. The stock
        ``walker.apply_action`` builds a fresh zero vector from the walker's
        *local* actuator indices, which would erase the other fly's controls
        and mis-address the goalie's actuators. Here we map each action
        entry to its global ``<walker>/<actuator>`` ctrl slot by name.
        """
        action = np.asarray(action, dtype=float)
        local_names = [
            a.name for a in walker.mjcf_model.find_all('actuator')
        ]
        for key in walker._action_indices.keys():
            if key == 'user':
                continue  # No MuJoCo actuator behind user actions.
            a_idx = walker._action_indices[key]
            c_idx = walker._ctrl_indices[key]
            if not c_idx or not a_idx:
                continue
            for ap, cl in zip(a_idx, c_idx):
                global_name = f'{walker.name}/{local_names[cl]}'
                try:
                    physics.named.data.ctrl[global_name] = action[ap]
                except KeyError:
                    pass
        walker._prev_action[:] = action

    def action_spec(self, physics):
        a_spec = self._attacker.get_action_spec(physics)
        if self._opponent_mode != 'self_play':
            return a_spec
        g_spec = self._goalie.get_action_spec(physics)
        minimum = np.concatenate([a_spec.minimum, g_spec.minimum])
        maximum = np.concatenate([a_spec.maximum, g_spec.maximum])
        return specs.BoundedArray(shape=(a_spec.shape[0] + g_spec.shape[0],),
                                  dtype=float,
                                  minimum=minimum,
                                  maximum=maximum,
                                  name='attacker+goalie')

    # -- Game logic helpers. --
    def _ball_state(self, physics):
        ball_qpos = np.asarray(physics.named.data.qpos['football'])
        ball_qvel = np.asarray(physics.named.data.qvel['football'])
        return ball_qpos[:3], ball_qvel[:3]

    def _walker_pos(self, physics, walker):
        pos, _ = walker.get_pose(physics)
        return np.asarray(pos)

    def _in_goal(self, ball_pos, side='east'):
        gx = (self._arena.east_goal_x if side == 'east' else
              self._arena.west_goal_x)
        crossed = (ball_pos[0] > gx) if side == 'east' else (ball_pos[0] < gx)
        return bool(crossed
                    and abs(ball_pos[1]) < self._arena.goal_width / 2.0
                    and ball_pos[2] < self._arena.goal_height)

    def get_reward_factors(self, physics):
        ball_pos, ball_vel = self._ball_state(physics)
        attacker_pos = self._walker_pos(physics, self._attacker)
        east_goal = np.array(
            [self._arena.east_goal_x, 0.0, self._arena.ball_radius])

        d_attacker_ball = np.linalg.norm(attacker_pos - ball_pos)
        approach = rewards.tolerance(d_attacker_ball,
                                     bounds=(0, 0.1),
                                     margin=2.0,
                                     sigmoid='linear',
                                     value_at_margin=0.0)

        d_ball_goal = np.linalg.norm(ball_pos - east_goal)
        ball_to_goal = rewards.tolerance(d_ball_goal,
                                         bounds=(0, 0.15),
                                         margin=3.0,
                                         sigmoid='linear',
                                         value_at_margin=0.0)

        # Ball velocity toward east goal (+x), scaled.
        kick = np.tanh(2.0 * float(ball_vel[0])) * 0.5 + 0.5

        factors = np.array([approach, ball_to_goal, kick])
        if self._scored:
            factors = np.append(factors, self._goal_bonus)
        if self._conceded:
            factors = np.append(factors, -self._concede_penalty)
        return factors

    def get_reward(self, physics):
        factors = self.get_reward_factors(physics)
        # Shaped dense part in [0, ~2.5] + sparse goal bonus.
        dense = float(0.4 * factors[0] + 1.0 * factors[1] +
                      0.6 * factors[2])
        sparse = float(np.sum(factors[3:])) if len(factors) > 3 else 0.0
        self._should_terminate = self.check_termination(physics)
        return dense + sparse

    def check_termination(self, physics):
        ball_pos, _ = self._ball_state(physics)
        if self._in_goal(ball_pos, 'east'):
            self._scored = True
            return True
        if self._in_goal(ball_pos, 'west'):
            self._conceded = True
            return True
        # Out of bounds (missed everything).
        if (abs(ball_pos[0]) > self._arena.field_length / 2.0 + 1.0
                or abs(ball_pos[1]) > self._arena.field_width / 2.0 + 1.0):
            return True
        # Flip / explosion guards.
        try:
            att_vel = np.linalg.norm(
                self._attacker.observables.velocimeter(physics))
            att_ang = np.linalg.norm(
                self._attacker.observables.gyro(physics))
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
            return (np.asarray(
                physics.named.data.qpos['football'][:3]) -
                    self._walker_pos(physics, self._attacker))

        return observable.Generic(get_vec)

    @composer.observable
    def ball_to_east_goal(self):
        def get_vec(physics):
            ball = np.asarray(
                physics.named.data.qpos['football'][:3])
            return (np.array([
                self._arena.east_goal_x, 0.0,
                self._arena.ball_radius,
            ]) - ball)

        return observable.Generic(get_vec)

    @composer.observable
    def goalie_to_ball(self):
        def get_vec(physics):
            return (np.asarray(
                physics.named.data.qpos['football'][:3]) -
                    self._walker_pos(physics, self._goalie))

        return observable.Generic(get_vec)
