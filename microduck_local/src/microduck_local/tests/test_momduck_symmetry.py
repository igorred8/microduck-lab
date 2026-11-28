"""Momduck's YZ-plane mirror map and its train-walk wiring.

These tests pin the observation/action algebra and checkpoint loading without
compiling the MJCF. The full MuJoCo kinematic check still belongs on the
machine that has Momduck's meshes.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from microduck_local import contract as C
from microduck_local import symmetry as S
from microduck_local import train as T
from microduck_local.robots import momduck_symmetry as M


def test_momduck_observation_signs_use_the_xz_reflection_plane():
    np.testing.assert_array_equal(M.MOMDUCK_OBS_SIGN[0:3], [-1, 1, -1])
    np.testing.assert_array_equal(M.MOMDUCK_OBS_SIGN[3:6], [1, -1, 1])
    np.testing.assert_array_equal(M.MOMDUCK_OBS_SIGN[48:51], [1, -1, -1])
    np.testing.assert_array_equal(M.MOMDUCK_OBS_SIGN[51:55], [1, 1, -1, -1])
    np.testing.assert_array_equal(
        M.MOMDUCK_OBS_SIGN[55:61], [1, -1, 1, -1, 1, -1])
    for base in (6, 20, 34):
        np.testing.assert_array_equal(
            M.MOMDUCK_OBS_SIGN[base:base + 14], S.JOINT_SIGN)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_momduck_mirror_is_an_involution(backend):
    rng = np.random.default_rng(40)
    obs = rng.normal(size=(5, C.OBS_DIM)).astype(np.float32)
    action = rng.normal(size=(5, C.NUM_JOINTS)).astype(np.float32)
    if backend == "torch":
        obs, action = torch.as_tensor(obs), torch.as_tensor(action)
        assert torch.equal(M.momduck_mirror_obs(M.momduck_mirror_obs(obs)), obs)
        assert torch.equal(
            M.momduck_mirror_action(M.momduck_mirror_action(action)), action)
    else:
        np.testing.assert_array_equal(
            M.momduck_mirror_obs(M.momduck_mirror_obs(obs)), obs)
        np.testing.assert_array_equal(
            M.momduck_mirror_action(M.momduck_mirror_action(action)), action)


def test_momduck_action_swaps_left_and_right_with_the_contract_signs():
    action = np.arange(1, C.NUM_JOINTS + 1, dtype=np.float32)
    expected = action[S.JOINT_PERM] * S.JOINT_SIGN
    np.testing.assert_array_equal(M.momduck_mirror_action(action), expected)
    np.testing.assert_allclose(
        M.momduck_mirror_action(C.DEFAULT_POSE), C.DEFAULT_POSE, atol=1e-7)


def test_momduck_normalized_mirror_goes_through_raw_observation_space():
    model = object.__new__(M.MomduckSymmetryPPO)
    rng = np.random.default_rng(41)
    raw = torch.as_tensor(rng.normal(size=(8, C.OBS_DIM)).astype(np.float32))
    mean = torch.as_tensor(rng.normal(size=C.OBS_DIM).astype(np.float32))
    std = torch.as_tensor(
        rng.uniform(0.5, 2.0, size=C.OBS_DIM).astype(np.float32))
    normalized = (raw - mean) / std

    got = model._mirrored_observation(normalized, (mean, std))
    want = (M.momduck_mirror_obs(raw) - mean) / std
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(M.momduck_mirror_obs(normalized), want, atol=1e-3)


def test_microduck_default_hooks_are_unchanged():
    model = object.__new__(S.SymmetryPPO)
    rng = np.random.default_rng(42)
    obs = torch.as_tensor(rng.normal(size=(3, C.OBS_DIM)).astype(np.float32))
    action = torch.as_tensor(rng.normal(size=(3, C.NUM_JOINTS)).astype(np.float32))
    assert torch.equal(model._mirror_observation(obs), S.mirror_obs(obs))
    assert torch.equal(model._mirror_action(action), S.mirror_action(action))


def test_train_selects_momduck_ppo_and_records_the_active_configuration():
    cls, kwargs = T.ppo_class_and_kwargs("momduck")
    assert cls is M.MomduckSymmetryPPO
    assert kwargs == {
        "symmetry_coef": 1.0,
        "hip_yaw_action_coef": 0.0,
        "hip_yaw_action_ramp": 0.5,
        "hip_yaw_action_limit_deg": 25.0,
    }
    stock_cls, stock_kwargs = T.ppo_class_and_kwargs("microduck")
    assert stock_cls is PPO and stock_kwargs == {}

    metadata = T.ppo_run_metadata(SimpleNamespace(
        symmetry_coef=0.5, symmetry_augment=False))
    # SimpleNamespace's type name is intentional: this checks the actual model,
    # rather than repeating what the CLI requested.
    assert metadata == {
        "ppo_class": "SimpleNamespace",
        "symmetry_coef": 0.5,
        "symmetry_augment": False,
        "hip_yaw_action_coef": 0.0,
        "hip_yaw_action_ramp": 0.0,
        "hip_yaw_action_limit_deg": 0.0,
    }


def test_fresh_momduck_explores_yaw_at_ten_degrees_only():
    venv = DummyVecEnv([_ContractEnv])
    model = PPO(
        S.FastActorCriticPolicy, venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[]), log_std_init=0.0),
        n_steps=8, batch_size=8, n_epochs=1, device="cpu", seed=0,
    )
    before = model.policy.log_std.detach().clone()
    T._set_fresh_exploration_std(model, "microduck")
    torch.testing.assert_close(model.policy.log_std.detach(), before)
    T._set_fresh_exploration_std(model, "momduck")
    std_deg = model.policy.log_std.detach().exp().rad2deg()
    torch.testing.assert_close(
        std_deg[[0, 9]], torch.tensor([10.0, 10.0]), atol=1e-5, rtol=0)
    other = [i for i in range(C.NUM_JOINTS) if i not in (0, 9)]
    torch.testing.assert_close(
        std_deg[other], torch.full((len(other),), float(np.degrees(1.0))),
        atol=1e-5, rtol=0)
    venv.close()


class _ContractEnv(gym.Env):
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, (C.OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(
        -4.0, 4.0, (C.NUM_JOINTS,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(C.OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return (np.zeros(C.OBS_DIM, dtype=np.float32), 0.0,
                False, False, {})


def test_stock_donor_loads_as_momduck_symmetry_ppo_without_changing_weights(tmp_path):
    """v6 was saved by stock PPO; subclass loading must preserve its policy."""
    venv = DummyVecEnv([_ContractEnv])
    donor = PPO(
        S.FastActorCriticPolicy, venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        n_steps=8, batch_size=8, n_epochs=1, device="cpu", seed=0,
    )
    donor.save(str(tmp_path / "model"))
    obs = np.random.default_rng(43).normal(
        size=(2, C.OBS_DIM)).astype(np.float32)
    expected = donor.predict(obs, deterministic=True)[0]

    cls, kwargs = T.ppo_class_and_kwargs("momduck")
    loaded = cls.load(
        str(tmp_path / "model"), env=venv, device="cpu",
        custom_objects={"policy_class": S.FastActorCriticPolicy},
        **kwargs,
    )
    assert isinstance(loaded, M.MomduckSymmetryPPO)
    assert loaded.symmetry_coef == M.MOMDUCK_SYMMETRY_COEF == 1.0
    assert loaded.symmetry_augment is False
    np.testing.assert_allclose(
        loaded.predict(obs, deterministic=True)[0], expected, atol=1e-6)
    assert T.ppo_run_metadata(loaded) == {
        "ppo_class": "MomduckSymmetryPPO",
        "symmetry_coef": 1.0,
        "symmetry_augment": False,
        "hip_yaw_action_coef": 0.0,
        "hip_yaw_action_ramp": 0.5,
        "hip_yaw_action_limit_deg": 25.0,
    }
    venv.close()


def test_momduck_learn_restarts_the_local_ramp_for_every_stage():
    venv = DummyVecEnv([_ContractEnv])
    model = M.MomduckSymmetryPPO(
        S.FastActorCriticPolicy, venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        n_steps=8, batch_size=8, n_epochs=1, device="cpu", seed=0,
        hip_yaw_action_coef=0.1,
    )
    model.num_timesteps = 14_089_000  # donor steps retained by SB3
    model.learn(total_timesteps=8, reset_num_timesteps=False)
    assert model._hip_yaw_run_start_steps == 14_089_000
    assert model._hip_yaw_run_total_steps == 8

    model.num_timesteps = 20_000_000
    model.learn(total_timesteps=16, reset_num_timesteps=False)
    assert model._hip_yaw_run_start_steps == 20_000_000
    assert model._hip_yaw_run_total_steps == 16
    venv.close()


def _hip_yaw_actor_test_model():
    model = object.__new__(M.MomduckSymmetryPPO)
    model.hip_yaw_action_coef = 0.05
    model.hip_yaw_action_ramp = 0.5
    model.hip_yaw_action_limit_deg = 25.0
    # Simulate a warm start: SB3's global progress is already exhausted, while
    # this run's local ramp must still begin at zero.
    model.num_timesteps = 14_089_000
    model._hip_yaw_run_start_steps = 14_089_000
    model._hip_yaw_run_total_steps = 1_000_000
    model._current_progress_remaining = 0.0
    return model


def test_momduck_hip_yaw_actor_hinge_ramps_and_charges_only_excess():
    model = _hip_yaw_actor_test_model()
    mean = torch.zeros((2, C.NUM_JOINTS), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        mean[:, 0] = np.radians(40.0)   # 15 degrees outside the safe range
        mean[:, 9] = np.radians(-20.0)  # safe: no neutral-angle pressure
        mean[:, 1:9] = 10.0  # other joints must not affect this regularizer
        mean[:, 10:] = -10.0

    # No new actor pressure at the donor boundary.
    assert model._policy_aux_losses(mean) == {}
    assert model._hip_yaw_action_scale() == 0.0

    # Halfway through the ramp the weighted excess MSE is exactly halved.
    # Global SB3 progress remains exhausted and must not affect this schedule.
    model.num_timesteps += 250_000
    half = model._policy_aux_losses(mean)["hip_yaw_target_limit"]
    excess = torch.relu(torch.abs(mean[:, [0, 9]]) - np.radians(25.0))
    raw = (excess ** 2).mean()
    torch.testing.assert_close(half, 0.05 * 0.5 * raw)

    model.num_timesteps += 250_000
    full = model._policy_aux_losses(mean)["hip_yaw_target_limit"]
    torch.testing.assert_close(full, 0.05 * raw)
    model.policy = None  # the loss must depend only on the supplied actor mean
    full.backward()
    grad = mean.grad
    assert grad[0, 0].item() != 0.0
    assert grad[0, 9].item() == 0.0  # -20 degrees is inside the hinge
    other = [i for i in range(C.NUM_JOINTS) if i not in (0, 9)]
    torch.testing.assert_close(grad[:, other], torch.zeros_like(grad[:, other]))

    model.hip_yaw_action_coef = 0.0
    assert model._policy_aux_losses(mean) == {}


def test_momduck_hip_yaw_actor_hinge_is_flat_inside_and_at_limits():
    model = _hip_yaw_actor_test_model()
    model.num_timesteps += 500_000
    mean = torch.zeros((2, C.NUM_JOINTS), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        mean[:, 0] = torch.tensor([-25.0, 24.0]).deg2rad()
        mean[:, 9] = torch.tensor([25.0, -24.0]).deg2rad()
        mean[:, 1] = 2.0

    loss = model._policy_aux_losses(mean)["hip_yaw_target_limit"]
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(mean.grad, torch.zeros_like(mean.grad))


def test_momduck_actor_metrics_separate_mean_excess_and_exploration_std():
    model = _hip_yaw_actor_test_model()
    model.num_timesteps += 250_000
    log_std = torch.zeros(C.NUM_JOINTS)
    log_std[0] = np.log(0.1)
    log_std[9] = np.log(0.2)
    model.policy = SimpleNamespace(log_std=log_std)
    mean = torch.zeros((2, C.NUM_JOINTS), dtype=torch.float32)
    mean[:, 0] = torch.tensor([30.0, -40.0]).deg2rad()
    mean[:, 9] = torch.tensor([-20.0, 25.0]).deg2rad()

    metrics = model._policy_aux_metrics(mean)
    assert set(metrics) == {
        "actor_hip_yaw_mean_left_deg",
        "actor_hip_yaw_mean_right_deg",
        "actor_hip_yaw_excess_left_deg",
        "actor_hip_yaw_excess_right_deg",
        "actor_hip_yaw_std_left_deg",
        "actor_hip_yaw_std_right_deg",
        "actor_hip_yaw_ramp_scale",
    }
    torch.testing.assert_close(metrics["actor_hip_yaw_mean_left_deg"],
                               torch.tensor(0.0))
    torch.testing.assert_close(metrics["actor_hip_yaw_mean_right_deg"],
                               torch.tensor(0.0))
    torch.testing.assert_close(metrics["actor_hip_yaw_excess_left_deg"],
                               torch.tensor(25.0))
    torch.testing.assert_close(metrics["actor_hip_yaw_excess_right_deg"],
                               torch.tensor(12.5))
    torch.testing.assert_close(metrics["actor_hip_yaw_std_left_deg"],
                               torch.rad2deg(torch.tensor(0.1)))
    torch.testing.assert_close(metrics["actor_hip_yaw_std_right_deg"],
                               torch.rad2deg(torch.tensor(0.2)))
    torch.testing.assert_close(metrics["actor_hip_yaw_ramp_scale"],
                               torch.tensor(0.5))
    assert all(not value.requires_grad for value in metrics.values())
