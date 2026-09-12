"""Test fly-vs-fly football environment."""

import numpy as np

from flybody.fly_envs import football_vs


def test_football_static_mode_runs():
    env = football_vs(opponent_mode='static', time_limit=1.0)
    assert env.action_spec().shape == (59,)
    ts = env.reset()
    assert 'attacker/ball_pos' in ts.observation
    assert 'attacker/ball_vel' in ts.observation
    assert 'attacker/attacker_to_ball' in ts.observation
    assert 'attacker/ball_to_east_goal' in ts.observation
    for _ in range(50):
        action = np.random.uniform(-0.5, 0.5, 59)
        ts = env.step(action)
        assert np.isfinite(ts.reward)
    # Both flies + ball present in physics.
    body_names = [
        env.physics.model.id2name(i, 'body')
        for i in range(env.physics.model.nbody)
    ]
    assert any('attacker' in (n or '') for n in body_names)
    assert any('goalie' in (n or '') for n in body_names)


def test_football_self_play_action_dim():
    env = football_vs(opponent_mode='self_play', time_limit=1.0)
    assert env.action_spec().shape == (118,)
    ts = env.reset()
    n = env.action_spec().shape[0]
    for _ in range(20):
        action = np.random.uniform(-0.3, 0.3, n)
        ts = env.step(action)
        assert np.isfinite(ts.reward)


def test_reward_shaping_penalizes_idleness():
    import numpy as np
    env = football_vs(opponent_mode='static', time_limit=10.0)
    env.reset()
    task = env.task
    # Doing nothing at spawn must pay almost nothing.
    r_rest = float(task.get_reward(env.physics))
    assert r_rest < 0.5, r_rest
    # Attacker next to the ball, ball rolling toward the east goal.
    ball_pos = np.asarray(env.physics.named.data.qpos['football'][:3])
    physics = env.physics
    spawn_z = float(task._attacker.upright_pose.xpos[2])
    physics.bind(task._root_joints['attacker']).qpos = np.array(
        [ball_pos[0] - 0.2, ball_pos[1], spawn_z, 1, 0, 0, 0])
    qvel = np.zeros(6)
    qvel[0] = 2.0  # Ball flying toward +x (east goal).
    physics.named.data.qvel['football'] = qvel
    r_good = float(task.get_reward(physics))
    assert r_good > r_rest + 0.5, (r_rest, r_good)


def test_football_goal_detection():
    env = football_vs(opponent_mode='static', time_limit=10.0)
    env.reset()
    task = env.task
    # Teleport ball just past east goal line, inside mouth.
    gx = task._arena.east_goal_x
    env.physics.named.data.qpos['football'] = np.array(
        [gx + 0.05, 0.0, 0.1, 1, 0, 0, 0])
    assert task.check_termination(env.physics) is True
    assert task._scored is True
