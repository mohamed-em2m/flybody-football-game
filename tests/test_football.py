"""Test fly-vs-fly football environment."""

import numpy as np

from flybody.fly_envs import football_vs


def test_football_static_mode_runs():
    env = football_vs(opponent_mode="static", time_limit=1.0)
    assert env.action_spec().shape == (59,)
    ts = env.reset()
    assert "attacker/ball_pos" in ts.observation
    assert "attacker/ball_vel" in ts.observation
    assert "attacker/attacker_to_ball" in ts.observation
    assert "attacker/ball_to_east_goal" in ts.observation
    for _ in range(50):
        action = np.random.uniform(-0.5, 0.5, 59)
        ts = env.step(action)
        assert np.isfinite(ts.reward)
    # Both flies + ball present in physics.
    body_names = [
        env.physics.model.id2name(i, "body") for i in range(env.physics.model.nbody)
    ]
    assert any("attacker" in (n or "") for n in body_names)
    assert any("goalie" in (n or "") for n in body_names)


def test_football_self_play_action_dim():
    env = football_vs(opponent_mode="self_play", time_limit=1.0)
    assert env.action_spec().shape == (118,)
    ts = env.reset()
    n = env.action_spec().shape[0]
    for _ in range(20):
        action = np.random.uniform(-0.3, 0.3, n)
        ts = env.step(action)
        assert np.isfinite(ts.reward)


def test_reward_shaping_penalizes_idleness():
    import numpy as np

    env = football_vs(opponent_mode="static", time_limit=10.0)
    env.reset()
    task = env.task
    # Doing nothing at spawn must pay almost nothing.
    r_rest = float(task.get_reward(env.physics))
    assert r_rest < 0.5, r_rest
    # Attacker next to the ball, ball rolling toward the east goal.
    ball_pos = np.asarray(env.physics.named.data.qpos["football"][:3])
    physics = env.physics
    spawn_z = float(task._attacker.upright_pose.xpos[2])
    physics.bind(task._root_joints["attacker"]).qpos = np.array(
        [ball_pos[0] - 0.2, ball_pos[1], spawn_z, 1, 0, 0, 0]
    )
    qvel = np.zeros(6)
    qvel[0] = 2.0  # Ball flying toward +x (east goal).
    physics.named.data.qvel["football"] = qvel
    r_good = float(task.get_reward(physics))
    assert r_good > r_rest + 0.5, (r_rest, r_good)


def test_football_goal_detection():
    env = football_vs(opponent_mode="static", time_limit=10.0)
    env.reset()
    task = env.task
    # Teleport ball just past east goal line, inside mouth.
    gx = task._arena.east_goal_x
    env.physics.named.data.qpos["football"] = np.array(
        [gx + 0.05, 0.0, 0.1, 1, 0, 0, 0]
    )
    assert task.check_termination(env.physics) is True
    assert task._scored is True


def _bodies(env):
    return [
        env.physics.model.id2name(i, "body") for i in range(env.physics.model.nbody)
    ]


def test_football_2v2_build():
    env = football_vs(opponent_mode="static", time_limit=1.0, n_per_team=2)
    assert env.action_spec().shape == (59,)  # west_0 only.
    env.reset()
    names = _bodies(env)
    for fly in ("west_0", "west_1", "east_0", "east_1"):
        assert any(fly in (n or "") for n in names), fly
    # Team color materials exist.
    mats = [
        env.physics.model.id2name(i, "material") for i in range(env.physics.model.nmat)
    ]
    assert any("west_0_jersey" in (m or "") for m in mats)
    assert any("east_0_jersey" in (m or "") for m in mats)
    # Legacy aliases still point at the lead flies.
    assert env.task.attacker.name == "west_0"
    assert env.task.goalie.name == "east_0"
    for _ in range(20):
        ts = env.step(np.random.uniform(-0.3, 0.3, 59))
        assert np.isfinite(ts.reward)


def test_football_2v2_self_play_dims():
    env = football_vs(opponent_mode="self_play", time_limit=1.0, n_per_team=2)
    assert env.action_spec().shape == (236,)  # 4 flies x 59.
    env.reset()
    for _ in range(10):
        ts = env.step(np.random.uniform(-0.2, 0.2, 236))
        assert np.isfinite(ts.reward)
        sides = env.task.get_side_rewards(env.physics)
        assert set(sides) == {"west", "east"}
        assert np.isfinite(sides["west"]) and np.isfinite(sides["east"])


def test_side_rewards_symmetric():
    env = football_vs(opponent_mode="static", time_limit=10.0)
    env.reset()
    task = env.task
    z = task._arena.ball_radius + 0.01
    # Ball near east goal, rolling east: good for west, bad for east.
    env.physics.named.data.qpos["football"] = np.array([1.5, 0.0, z, 1, 0, 0, 0])
    qvel = np.zeros(6)
    qvel[0] = 2.0
    env.physics.named.data.qvel["football"] = qvel
    sides = task.get_side_rewards(env.physics)
    assert sides["west"] > sides["east"] + 1.0, sides
    # Mirror: ball near west goal, rolling west.
    env.physics.named.data.qpos["football"] = np.array([-1.5, 0.0, z, 1, 0, 0, 0])
    qvel = np.zeros(6)
    qvel[0] = -2.0
    env.physics.named.data.qvel["football"] = qvel
    sides = task.get_side_rewards(env.physics)
    assert sides["east"] > sides["west"] + 1.0, sides


def test_possession_observable():
    env = football_vs(opponent_mode="static", time_limit=1.0, extended_obs=True)
    env.reset()
    task = env.task
    physics = env.physics
    att_pos = task._walker_pos(physics, task.attacker)
    z = task._arena.ball_radius + 0.01
    physics.named.data.qpos["football"] = np.array(
        [att_pos[0] + 0.1, att_pos[1], z, 1, 0, 0, 0]
    )
    physics.named.data.qvel["football"] = np.zeros(6)
    ts = env.step(np.zeros(59))
    poss = np.asarray(ts.observation["attacker/possession"])
    assert poss.tolist() == [1.0, 0.0, 0.0], poss
    assert task.possession_side == "west"


def _set_ball(physics, x, y, z, vx=0.0, vy=0.0):
    physics.named.data.qpos["football"] = np.array([x, y, z, 1, 0, 0, 0])
    qvel = np.zeros(6)
    qvel[0], qvel[1] = vx, vy
    physics.named.data.qvel["football"] = qvel


def test_interception_event():
    env = football_vs(
        opponent_mode="static",
        time_limit=10.0,
        possession_min_hold=2,
        interception_cooldown=0,
    )
    env.reset()
    task = env.task
    physics = env.physics
    rs = np.random.RandomState(1)
    z = task._arena.ball_radius + 0.01
    zero = np.zeros(59)
    # West holds the ball for 3 steps.
    att_pos = task._walker_pos(physics, task.attacker)
    _set_ball(physics, att_pos[0] + 0.1, att_pos[1], z)
    for _ in range(3):
        task.before_step(physics, zero, rs)
    assert task.possession_side == "west"
    # Ball jumps to the goalie: turnover after a real hold.
    goal_pos = task._walker_pos(physics, task.goalie)
    _set_ball(physics, goal_pos[0] + 0.1, goal_pos[1], z)
    task.before_step(physics, zero, rs)
    assert task.possession_side == "east"
    assert task._pending_bonus["east"] == task._interception_bonus
    name, side, _ = task.last_event_info
    assert (name, side) == ("interception", "east")
    sides = task.get_side_rewards(physics)
    # Dense shaping still favors west here (ball parked by the east goal),
    # but east's total must contain the +1.0 interception bonus on top of
    # its (negative) dense part.
    assert sides["east"] > 0.5, sides


def test_pass_and_shot_events():
    env = football_vs(
        opponent_mode="static", time_limit=10.0, n_per_team=2, pass_window=40
    )
    env.reset()
    task = env.task
    physics = env.physics
    rs = np.random.RandomState(2)
    z = task._arena.ball_radius + 0.01
    zero = np.zeros(59)
    w0_pos = task._walker_pos(physics, task.west[0])
    w1_pos = task._walker_pos(physics, task.west[1])
    # Settle: ball resting next to west_0.
    _set_ball(physics, w0_pos[0] + 0.1, w0_pos[1], z)
    task.before_step(physics, zero, rs)
    # Sideways kick (not a shot: perpendicular to goal direction).
    _set_ball(physics, w0_pos[0] + 0.1, w0_pos[1], z, vx=0.0, vy=2.5)
    task.before_step(physics, zero, rs)
    assert task._last_kick is not None
    assert task._last_kick[0] == "west"
    assert task._pending_bonus["west"] == 0.0  # No shot, no save.
    # Teammate collects the moving ball: completed pass.
    _set_ball(physics, w1_pos[0] + 0.1, w1_pos[1], z, vx=0.0, vy=2.5)
    task.before_step(physics, zero, rs)
    assert task._pending_bonus["west"] == task._pass_bonus
    name, side, _ = task.last_event_info
    assert (name, side) == ("pass", "west")
    # Holding longer must not pay the pass again (no farming).
    task.before_step(physics, zero, rs)
    assert task._pending_bonus["west"] == 0.0


def test_out_of_bounds_penalty():
    env = football_vs(opponent_mode="static", time_limit=10.0)
    env.reset()
    task = env.task
    physics = env.physics
    spawn_z = float(task._attacker.upright_pose.xpos[2])
    # Baseline with everyone on the pitch.
    base = task.get_side_rewards(physics)
    # Teleport the west attacker far outside the pitch.
    hx = task._arena.field_length / 2.0 + task._out_of_bounds_margin + 0.5
    physics.bind(task._root_joints["attacker"]).qpos = np.array(
        [hx, 0.0, spawn_z, 1, 0, 0, 0]
    )
    sides = task.get_side_rewards(physics)
    assert task._out_of_bounds_frac(physics) == {"west": 1.0, "east": 0.0}
    assert sides["west"] < base["west"] - 0.5, (base, sides)
    assert sides["east"] == base["east"]
    # East pays too when its own fly leaves the pitch.
    hy = task._arena.field_width / 2.0 + task._out_of_bounds_margin + 0.5
    physics.bind(task._root_joints["goalie"]).qpos = np.array(
        [0.0, hy, spawn_z, 1, 0, 0, 0]
    )
    sides2 = task.get_side_rewards(physics)
    assert sides2["east"] < base["east"] - 0.5, (base, sides2)
    assert sides2["west"] == sides["west"]
    # Legacy scalar reward carries the west penalty as well.
    r_oob = float(task.get_reward(physics))
    physics.bind(task._root_joints["attacker"]).qpos = np.array(
        [0.0, 0.0, spawn_z, 1, 0, 0, 0]
    )
    assert float(task.get_reward(physics)) > r_oob + 0.5
