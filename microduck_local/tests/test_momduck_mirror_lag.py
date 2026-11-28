"""Planted-control test for the half-period mirror term: a held asymmetric pose
is charged after the lag, the mirror-symmetric STAND is free."""
import numpy as np

from microduck_local.robots.momduck_env import MomduckWalkEnv


def _env():
    return MomduckWalkEnv(actuator_force="xml", obs_noise=False,
                          domain_rand=False, action_delay=False, push_robot=False)


def test_mirror_lag_charged_on_held_asymmetry_and_free_on_stand():
    env = _env()
    env.reset(seed=0)
    for _ in range(env.MIRROR_LAG_STEPS + 1):
        env.step(np.zeros(14, dtype=np.float32))
    _, terms = env._compute_reward()
    # STAND itself is not perfectly mirror-symmetric, so require only that it
    # is much cheaper than the planted asymmetric pose.
    assert abs(terms["mirror_lag_penalty"]) < 0.01, terms["mirror_lag_penalty"]

    env.reset(seed=0)
    a = np.zeros(14, dtype=np.float32)
    a[0] = 0.3  # hold left_hip_yaw away from the mirror-symmetric default
    for _ in range(env.MIRROR_LAG_STEPS + 1):
        env.step(a)
    _, terms = env._compute_reward()
    assert terms["mirror_lag_penalty"] < -0.01, terms["mirror_lag_penalty"]
    env.close()
