"""Planted-control test: backward sole normal on a NEW contact is charged once,
and only there — flat contact and already-planted soles are free."""
import numpy as np

from microduck_local.robots.momduck_env import MomduckWalkEnv


def _env():
    return MomduckWalkEnv(actuator_force="xml", obs_noise=False,
                          domain_rand=False, action_delay=False, push_robot=False)


def test_heel_strike_charged_once_on_new_contact_only():
    env = _env()
    env.reset(seed=0)
    # Override BEFORE the first contact evaluation: the sole normal is backward.
    gid = env.foot_geoms["left"]
    R = env.data.geom_xmat[gid].reshape(3, 3)
    env._sole_normal_local["left"] = R.T @ np.array([-0.2, 0.0, 0.98])
    reward, terms = env._compute_reward()
    expected = -env.W_HEEL_STRIKE * (0.2 - env.HEEL_STRIKE_TOL)
    assert abs(terms["heel_strike_penalty"] - expected) < 1e-6, (
        terms["heel_strike_penalty"], expected)
    # Same sole still planted: NOT a new contact, so no second charge.
    _, terms2 = env._compute_reward()
    assert terms2["heel_strike_penalty"] == 0.0, terms2["heel_strike_penalty"]

    # Fresh reset, no override: flat STAND must be free.
    env.reset(seed=0)
    _, terms3 = env._compute_reward()
    assert terms3["heel_strike_penalty"] == 0.0, terms3["heel_strike_penalty"]
    env.close()
