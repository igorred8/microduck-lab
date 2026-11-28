"""Planted-control test for the speed-stability term: constant forward speed is
free, a jerky speed is charged, and standing still is not charged."""
import numpy as np

from microduck_local.robots.momduck_env import MomduckWalkEnv


def _env():
    return MomduckWalkEnv(actuator_force="xml", obs_noise=False,
                          domain_rand=False, action_delay=False, push_robot=False)


def test_speed_stability_constant_free_jerky_charged_standing_not_charged():
    env = _env()
    env.reset(seed=0)
    # Constant forward speed: variance ~zero -> charge is ~zero (the current
    # standing velocity is appended to the window by _compute_reward itself).
    env._vx_hist.extend([0.1] * (env.SPEED_STABILITY_WINDOW - 1))
    _, terms = env._compute_reward()
    assert abs(terms["speed_stability_penalty"]) < 0.001, terms["speed_stability_penalty"]

    # Alternating speeds: nonzero variance -> charge.
    env._vx_hist.clear()
    env._vx_hist.extend([0.0, 0.2] * (env.SPEED_STABILITY_WINDOW // 2) + [0.0])
    _, terms = env._compute_reward()
    assert terms["speed_stability_penalty"] < -0.001, terms["speed_stability_penalty"]

    # Standing still: variance zero, but also below the motion gate -> not charged.
    env._vx_hist.clear()
    env._vx_hist.extend([0.0] * env.SPEED_STABILITY_WINDOW)
    _, terms = env._compute_reward()
    assert terms["speed_stability_penalty"] == 0.0, terms["speed_stability_penalty"]
    env.close()
