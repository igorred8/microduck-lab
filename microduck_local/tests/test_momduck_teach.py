"""Momduck teach recipes and environment-owned locomotion curricula."""

import json
from types import SimpleNamespace

import numpy as np
from test_lab import fake_popen  # noqa: F401 — fixture: records trainer launches

from microduck_local import behaviors as B
from microduck_local import train as T
from microduck_local import viz_server as V
from microduck_local.lab import robots as lab_robots
from microduck_local.robots import momduck_env as E
from microduck_local.robots.momduck_curriculum import MomduckRunCurriculum


def test_momduck_suggestion_matches_the_momduck_recipe(monkeypatch):
    suggestions = lab_robots.teach_suggestions("momduck")
    assert [s["behavior"] for s in suggestions] == [
        "momduck_stand", "momduck_forward", "momduck_walk", "momduck_run"]
    monkeypatch.setattr(lab_robots, "available_robots", lambda: [{
        "id": "momduck", "label": "Momduck", "noun": "momduck",
        "kind": "legged", "ready": True, "animate": True,
    }])
    entries = {entry["id"]: entry for entry in lab_robots.robot_entries()}
    assert entries["momduck"]["teach"] == suggestions
    behavior, clip = V.match_teach_text(suggestions[0]["text"], "momduck")
    assert behavior is B.BEHAVIORS["momduck_stand"]
    assert clip is None
    assert V.match_teach_text("momduck_stand", "microduck") == (None, None)
    assert V.match_teach_text("run forward", "momduck")[0] is B.BEHAVIORS["momduck_run"]
    assert V.match_teach_text("momduck_run", "microduck") == (None, None)


def test_momduck_recipe_builds_the_nominal_xml_stand_training_kwargs():
    behavior = B.BEHAVIORS["momduck_stand"]
    args = T.parse_args([
        *behavior.trainer[2:],
        "--envs", "12",
        "--steps", "1000000",
        "--run-name", "momduck-stand-v1",
    ])
    assert (args.robot, args.task, args.actuator) == ("momduck", "stand", "xml")
    assert (args.envs, args.steps, args.run_name) == (12, 1000000, "momduck-stand-v1")
    assert args.lr == 0.0003
    assert T.env_class(args.robot, args.task).__name__ == "MomduckStandEnv"
    assert T.is_pinned_command(args.task)
    kwargs = T.env_kwargs_from_args(args)
    assert kwargs == {
        "domain_rand": False,
        "obs_noise": False,
        "actuator_force": "xml",
        "max_episode_s": 20.0,
        "action_delay": False,
        "push_robot": False,
    }


def test_momduck_forward_warm_start_uses_clean_xml_and_its_own_env():
    args = T.parse_args([
        "--robot", "momduck",
        "--task", "forward",
        "--actuator", "xml",
        "--init-from", "runs/teach-momduck_stand-example-s1",
        "--envs", "12",
        "--steps", "2000000",
        "--lr", "0.0001",
        "--run-name", "momduck-forward-v1",
    ])
    assert T.env_class(args.robot, args.task).__name__ == "MomduckForwardEnv"
    assert not T.is_pinned_command(args.task)
    assert args.freeze_obs_norm
    assert T.env_kwargs_from_args(args) == {
        "domain_rand": False,
        "obs_noise": False,
        "actuator_force": "xml",
        "max_episode_s": 20.0,
        "action_delay": False,
        "push_robot": False,
    }


def test_momduck_forward_command_is_always_straight_ahead():
    cls = T.env_class("momduck", "forward")
    env = object.__new__(cls)  # command logic does not need to compile the MJCF
    env.twist_cmd = np.zeros(3)
    env.head_cmd = np.zeros(4)
    env.body_cmd = np.zeros(6)
    for _ in range(3):
        env._sample_commands()
        np.testing.assert_allclose(env.twist_cmd, (0.15, 0.0, 0.0))
        np.testing.assert_allclose(env.head_cmd, 0.0)
        np.testing.assert_allclose(env.body_cmd, 0.0)
    assert env.FORWARD_FALL_HEIGHT == 0.14


def test_momduck_velocity_adapter_maps_x_forward_and_plus_y_left(monkeypatch):
    env = object.__new__(E.MomduckWalkEnv)
    monkeypatch.setattr(E.MicroduckWalkEnv, "body_lin_vel",
                        lambda self: np.array([0.10, 0.15, 0.02]))
    np.testing.assert_allclose(env.body_lin_vel(), (0.10, 0.15, 0.02))

    env._qvel_base = np.array([0.10, 0.15, 0.02])
    env._trunk_xmat = np.eye(3).reshape(-1)
    np.testing.assert_allclose(env.heading_lin_vel(), (0.10, 0.15, 0.02))


def test_momduck_forward_rejects_lateral_travel_and_rotated_heading(monkeypatch):
    monkeypatch.setattr(E.MomduckWalkEnv, "_compute_reward",
                        lambda self: (0.0, {"feet_air_time": 3.0}))
    monkeypatch.setattr(E.MomduckForwardEnv, "_walking_stance_terms",
                        lambda self: (0.0, 0.0, 0.0, 4.0))
    monkeypatch.setattr(E.MomduckForwardEnv, "_hip_roll_terms",
                        lambda self: (0.0, 0.0))
    monkeypatch.setattr(E.MomduckForwardEnv, "_hip_yaw_terms",
                        lambda self: (0.0, 0.0))
    monkeypatch.setattr(E.MomduckForwardEnv, "_head_yaw_roll_penalty",
                        lambda self: 0.0)
    monkeypatch.setattr(E.MomduckForwardEnv, "_swing_stride_reward",
                        lambda self: 3.0)
    env = object.__new__(E.MomduckForwardEnv)
    env.stand_z = 0.156
    env._trunk_xpos = np.array([0.0, 0.0, 0.156])
    env._spawn_forward = np.array([1.0, 0.0, 0.0])
    env._spawn_side = np.array([0.0, 1.0, 0.0])
    env._trunk_xmat = np.eye(3).reshape(-1)
    env._gyro = np.zeros(3)
    assert env.BODY_FORWARD == (1.0, 0.0, 0.0)

    env._qvel_base = np.array([0.15, 0.0, 0.0])
    _, straight = env._compute_reward()
    assert straight["forward_progress"] == 2.0
    assert straight["lateral_velocity_penalty"] == 0.0
    assert straight["heading_penalty"] == 0.0
    assert straight["yaw_rate_penalty"] == 0.0
    assert straight["track_lin_vel"] == 4.0
    assert straight["feet_air_time"] == 3.0
    assert straight["foot_clearance"] == 4.0
    assert straight["swing_stride"] == 3.0
    assert "hip_yaw_target_limit_penalty" not in straight

    # The old +Y assumption is lateral for the corrected MJCF and must not pay.
    env._qvel_base = np.array([0.0, 0.15, 0.0])
    _, old_y = env._compute_reward()
    assert old_y["forward_progress"] == 0.0
    assert old_y["lateral_velocity_penalty"] < 0.0
    assert old_y["heading_penalty"] == 0.0
    assert old_y["feet_air_time"] == 0.0
    assert old_y["foot_clearance"] == 0.0
    assert old_y["swing_stride"] == 0.0

    # Momduck's body +X points along world -X and the robot travels along
    # world -X. An initial-course reward must not call this forward tracking.
    env._trunk_xmat = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]).reshape(-1)
    env._qvel_base = np.array([-0.15, 0.0, 0.0])
    _, sideways = env._compute_reward()
    assert sideways["forward_progress"] == -2.0
    assert sideways["lateral_velocity_penalty"] == 0.0
    np.testing.assert_allclose(
        sideways["heading_penalty"], -12.0 * (np.pi / 2) ** 2)
    assert sideways["track_lin_vel"] < straight["track_lin_vel"]

    # A persistent turn is charged directly even before heading error grows.
    env._trunk_xmat = np.eye(3).reshape(-1)
    env._qvel_base = np.array([0.15, 0.0, 0.0])
    env._gyro = np.array([0.0, 0.0, 0.2])
    _, turning = env._compute_reward()
    np.testing.assert_allclose(turning["yaw_rate_penalty"], -0.48)
    assert turning["heading_penalty"] == 0.0


def test_momduck_forward_anchors_only_the_support_leg_to_stand():
    env = object.__new__(E.MomduckForwardEnv)
    rel = np.zeros(14, dtype=np.float32)
    rel[env._RIGHT_LEG_IDS] = 0.8  # a badly posed but swinging right leg
    env._joint_pos_rel = lambda: rel
    env._foot_contacts = lambda: {"left": True, "right": False}
    env.foot_geoms = {"left": 0, "right": 1}
    env.foot_flat_ref = {
        side: np.array([0.0, 0.0, -1.0]) for side in ("left", "right")
    }
    env.foot_yaw_axes = {side: (1, 1.0) for side in ("left", "right")}
    env.foot_forward_ref = {
        side: np.array([0.0, 1.0, 0.0]) for side in ("left", "right")
    }
    env.foot_stand_z = {"left": 0.0, "right": 0.0}
    env.data = SimpleNamespace(
        geom_xmat=np.tile(np.eye(3).reshape(-1), (2, 1)),
        geom_xpos=np.zeros((2, 3)),
    )

    # The left foot is loaded and nominal, so the right leg's large deviation
    # is free: it is the swing leg and must be allowed to make a stride.
    assert env._walking_stance_terms() == (0.0, 0.0, 0.0, 0.0)

    env._foot_contacts = lambda: {"left": True, "right": True}
    pose_penalty, flat_penalty, yaw_penalty, clearance = \
        env._walking_stance_terms()
    assert pose_penalty < 0.0
    assert flat_penalty == 0.0
    assert yaw_penalty == 0.0
    assert clearance == 0.0

    # The same right-leg deviation is charged as soon as that foot bears load.
    env._foot_contacts = lambda: {"left": False, "right": True}
    right_pose, right_flat, right_yaw, right_clearance = \
        env._walking_stance_terms()
    # With both feet loaded, the same bad leg contributes half the mean cost.
    np.testing.assert_allclose(right_pose, 2.0 * pose_penalty)
    assert right_flat == 0.0
    assert right_yaw == 0.0
    assert right_clearance == 0.0

    # Pure yaw leaves the sole perfectly flat. The v4 flatness-only test missed
    # exactly this toe-in failure; the new directional term must catch it.
    env.data.geom_xmat[1] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]).reshape(-1)
    _, yawed_flat, yawed_direction, _ = env._walking_stance_terms()
    assert yawed_flat == 0.0
    assert yawed_direction < 0.0

    # A tilted loaded sole remains an independent orientation error.
    env.data.geom_xmat[1] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]).reshape(-1)
    _, tilted_flat, _, _ = env._walking_stance_terms()
    assert tilted_flat < 0.0

    # Running must not inherit this walking-specific support-style prior.
    assert E.MomduckRunEnv.W_STANCE_POSE == 0.0
    assert E.MomduckRunEnv.W_STANCE_FLAT == 0.0
    assert E.MomduckRunEnv.W_STANCE_YAW == 0.0
    assert E.MomduckRunEnv.W_FOOT_CLEARANCE == 0.0


def test_momduck_forward_pays_swing_clearance_but_never_a_two_foot_hop():
    env = object.__new__(E.MomduckForwardEnv)
    env._joint_pos_rel = lambda: np.zeros(14, dtype=np.float32)
    env.foot_geoms = {"left": 0, "right": 1}
    env.foot_stand_z = {"left": 0.004, "right": 0.004}
    env.foot_flat_ref = {
        side: np.array([0.0, 0.0, -1.0]) for side in ("left", "right")
    }
    env.foot_yaw_axes = {side: (1, 1.0) for side in ("left", "right")}
    env.foot_forward_ref = {
        side: np.array([0.0, 1.0, 0.0]) for side in ("left", "right")
    }
    env.data = SimpleNamespace(
        geom_xmat=np.tile(np.eye(3).reshape(-1), (2, 1)),
        geom_xpos=np.array([[0.0, 0.0, 0.004], [0.0, 0.0, 0.004]]),
    )

    # A 1.5 cm lift already pays half the clearance term: there is no
    # 0.125-second air-time dead zone between shuffling and a real step.
    env.data.geom_xpos[1, 2] += 0.015
    env._foot_contacts = lambda: {"left": True, "right": False}
    *_, clearance = env._walking_stance_terms()
    np.testing.assert_allclose(clearance, 2.0)

    env.data.geom_xpos[1, 2] += 0.015
    *_, clearance = env._walking_stance_terms()
    assert clearance == 4.0

    # Both feet airborne is a hop/flight phase, not a walking swing: no pay.
    env._foot_contacts = lambda: {"left": False, "right": False}
    assert env._walking_stance_terms() == (0.0, 0.0, 0.0, 0.0)

    # A raised foot that has touched down no longer receives clearance pay.
    env._foot_contacts = lambda: {"left": True, "right": True}
    *_, clearance = env._walking_stance_terms()
    assert clearance == 0.0


def test_momduck_forward_rewards_longitudinal_swing_stride_not_a_hop():
    env = object.__new__(E.MomduckForwardEnv)
    env._spawn_forward = np.array([1.0, 0.0, 0.0])
    env.foot_geoms = {"left": 0, "right": 1}
    env.data = SimpleNamespace(geom_xpos=np.array([
        [0.0, 0.0, 0.004],
        [-0.04, 0.0, 0.034],
    ]))
    env._foot_contacts = lambda: {"left": True, "right": False}

    # The ramp starts 4 cm behind the support foot and has useful gradient
    # before the swing foot overtakes it; clearance alone had no such signal.
    assert env._swing_stride_reward() == 0.0
    env.data.geom_xpos[1, 0] = 0.0
    np.testing.assert_allclose(env._swing_stride_reward(), 1.5)
    env.data.geom_xpos[1, 0] = 0.04
    np.testing.assert_allclose(env._swing_stride_reward(), 3.0)

    env._foot_contacts = lambda: {"left": True, "right": True}
    assert env._swing_stride_reward() == 0.0
    env._foot_contacts = lambda: {"left": False, "right": False}
    assert env._swing_stride_reward() == 0.0
    assert E.MomduckRunEnv.W_SWING_STRIDE == 0.0


def test_momduck_forward_prices_hip_roll_near_limits_even_on_swing_leg():
    env = object.__new__(E.MomduckForwardEnv)
    rel = np.zeros(14, dtype=np.float32)
    env._joint_pos_rel = lambda: rel
    assert env._hip_roll_terms() == (0.0, 0.0)

    # STAND left=-5deg and range [-25,+20]deg: relative -0.349 rad reaches the
    # lower mechanical limit. Unlike sagittal stride motion, this is always bad,
    # including when that leg is swinging.
    rel[1] = -0.349066
    centre, limit = env._hip_roll_terms()
    assert centre < 0.0
    assert limit < 0.0

    # A small balance correction stays inside the free band and pays no barrier.
    rel[1] = -0.10
    small_centre, small_limit = env._hip_roll_terms()
    assert small_centre < 0.0
    assert small_limit == 0.0
    assert abs(small_centre) < abs(centre)

    # The running recipe does not inherit this walking-posture prior.
    assert E.MomduckRunEnv.W_HIP_ROLL == 0.0
    assert E.MomduckRunEnv.W_HIP_ROLL_LIMIT == 0.0


def test_momduck_forward_rejects_chaplin_hip_yaw_on_either_leg():
    env = object.__new__(E.MomduckForwardEnv)
    env._lifetime_steps = 0
    rel = np.zeros(14, dtype=np.float32)
    env._joint_pos_rel = lambda: rel

    # v13 is a diagnostic continuation from the last good v12 checkpoint. It
    # restores the stable v11 objective in full from step zero: no third ramp
    # and no return to the v12 weights that preceded the forward falls.
    assert env.W_HIP_YAW == 12.0
    assert env.W_HIP_YAW_LIMIT == 24.0
    assert env._hip_yaw_weight_scale() == 1.0
    assert env._hip_yaw_free_band() == 0.15
    assert env._hip_yaw_terms() == (0.0, 0.0)

    # A small steering correction remains inside the barrier-free band, but
    # the quadratic centre term still pulls it toward DEFAULT_POSE.
    rel[0] = 0.08
    small_centre, small_limit = env._hip_yaw_terms()
    assert small_centre < 0.0
    assert small_limit == 0.0

    # The RIGHT SWING hip at its known-good +25 degree limit is charged too.
    rel[0] = 0.0
    rel[9] = 0.436332
    centre, limit = env._hip_yaw_terms()
    yaw = rel[env._HIP_YAW_IDS].astype(float)
    excess = np.maximum(np.abs(yaw) - 0.15, 0.0)
    np.testing.assert_allclose(centre, -12.0 * float(np.mean(yaw ** 2)))
    np.testing.assert_allclose(limit, -24.0 * float(np.mean(excess ** 2)))
    assert abs(centre) > abs(small_centre)

    # Nothing changes as lifetime advances: this run collects measurements
    # under one fixed reward instead of confounding them with another ramp.
    env._lifetime_steps = env.HIP_YAW_RAMP_STEPS * 2
    assert env._hip_yaw_weight_scale() == 1.0
    assert env._hip_yaw_free_band() == 0.15
    np.testing.assert_allclose(env._hip_yaw_terms(), (centre, limit))

    rel[9] = 0.0
    assert E.MomduckRunEnv.W_HIP_YAW == 0.0
    assert E.MomduckRunEnv.W_HIP_YAW_LIMIT == 0.0


def test_momduck_forward_measures_target_excess_without_an_environment_penalty():
    env = object.__new__(E.MomduckForwardEnv)
    env._ctrl_whole = True
    env.default_pose = np.zeros(14, dtype=np.float32)
    limit = np.radians(25.0)
    env._hip_yaw_target_lo = np.array([-limit, -limit])
    env._hip_yaw_target_hi = np.array([limit, limit])
    target = np.zeros(14, dtype=np.float32)
    env.data = SimpleNamespace(ctrl=target)

    # v16 keeps the useful measurement but none of the failed v15 reward path.
    assert not hasattr(E.MomduckForwardEnv, "W_HIP_YAW_TARGET_LIMIT")
    assert not hasattr(E.MomduckForwardEnv, "_hip_yaw_target_limit_penalty")
    target[0] = -limit
    target[9] = limit
    np.testing.assert_allclose(env._hip_yaw_target_excess(), 0.0, atol=1e-8)

    # At -80/+30 degrees only 55/5 degrees outside the stops are measured.
    target[0] = np.radians(-80.0)
    target[9] = np.radians(30.0)
    excess = np.array([np.radians(55.0), np.radians(5.0)])
    np.testing.assert_allclose(env._hip_yaw_target_excess(), excess, atol=1e-8)


def test_momduck_forward_reports_measured_height_speed_and_hip_angles(monkeypatch):
    env = object.__new__(E.MomduckForwardEnv)
    env._trunk_xpos = np.array([0.0, 0.0, 0.148])
    env.stand_z = 0.156
    env._fall_height = 0.14
    env._spawn_forward = np.array([1.0, 0.0, 0.0])
    env._spawn_side = np.array([0.0, 1.0, 0.0])
    env._qvel_base = np.array([0.17, 0.01, 0.0])
    rel = np.zeros(14, dtype=np.float32)
    rel[0] = np.radians(12.0)
    rel[9] = np.radians(-14.0)
    rel[1] = np.radians(-5.0)
    rel[10] = np.radians(6.0)
    env._joint_pos_rel = lambda: rel
    env._ctrl_whole = True
    env.default_pose = np.zeros(14, dtype=np.float32)
    limit = np.radians(25.0)
    env._hip_yaw_target_lo = np.array([-limit, -limit])
    env._hip_yaw_target_hi = np.array([limit, limit])
    target = np.zeros(14, dtype=np.float32)
    target[0] = np.radians(30.0)
    target[9] = np.radians(-40.0)
    env.data = SimpleNamespace(ctrl=target)

    metrics = env._diagnostic_metrics()
    np.testing.assert_allclose(metrics["height_m"], 0.148)
    np.testing.assert_allclose(metrics["height_rel_stand_m"], -0.008)
    np.testing.assert_allclose(metrics["fall_margin_m"], 0.008)
    np.testing.assert_allclose(metrics["forward_speed_mps"], 0.17)
    np.testing.assert_allclose(metrics["lateral_speed_mps"], 0.01)
    np.testing.assert_allclose(metrics["hip_yaw_left_deg"], 12.0)
    np.testing.assert_allclose(metrics["hip_yaw_right_deg"], -14.0)
    np.testing.assert_allclose(metrics["hip_yaw_target_left_deg"], 30.0)
    np.testing.assert_allclose(metrics["hip_yaw_target_right_deg"], -40.0)
    np.testing.assert_allclose(metrics["hip_yaw_target_excess_left_deg"], 5.0, atol=1e-5)
    np.testing.assert_allclose(metrics["hip_yaw_target_excess_right_deg"], 15.0, atol=1e-5)
    np.testing.assert_allclose(metrics["hip_roll_left_deg"], -5.0)
    np.testing.assert_allclose(metrics["hip_roll_right_deg"], 6.0)

    # The normal Gym step publishes these as auxiliary info, outside both the
    # 61-observation contract and the reward-term scorecard.
    monkeypatch.setattr(
        E.MicroduckWalkEnv, "step",
        lambda self, action: ("obs", 0.0, False, False, {}))
    *_, info = env.step(np.zeros(14, dtype=np.float32))
    assert info["momduck_metrics"] == metrics


def test_momduck_training_progress_averages_diagnostic_metrics(tmp_path):
    from stable_baselines3.common.callbacks import BaseCallback

    callback = T._watch_callback_cls(BaseCallback)(
        tmp_path, total_steps=100, snap_steps=1000, obs_dim=61)
    callback.model = SimpleNamespace(
        num_timesteps=12,
        logger=SimpleNamespace(name_to_value={
            "train/actor_hip_yaw_mean_left_deg": -30.123456,
            "train/actor_hip_yaw_excess_left_deg": 5.0,
            "train/actor_hip_yaw_std_left_deg": 8.0,
            "train/actor_hip_yaw_ramp_scale": 0.5,
            "train/policy_gradient_loss": 123.0,
        }),
    )
    callback.locals = {"infos": [
        {"momduck_metrics": {
            "height_m": 0.15, "forward_speed_mps": 0.15,
            "hip_yaw_left_deg": 10.0,
        }},
        {"momduck_metrics": {
            "height_m": 0.13, "forward_speed_mps": 0.17,
            "hip_yaw_left_deg": 20.0,
        }},
    ]}
    assert callback._on_step()
    callback._on_rollout_end()

    line = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert line["metrics"] == {
        "height_m": 0.14,
        "forward_speed_mps": 0.16,
        "hip_yaw_left_deg": 15.0,
    }
    assert line["actor"] == {
        "actor_hip_yaw_mean_left_deg": -30.12346,
        "actor_hip_yaw_excess_left_deg": 5.0,
        "actor_hip_yaw_std_left_deg": 8.0,
        "actor_hip_yaw_ramp_scale": 0.5,
    }
    assert callback.metric_samples == 0
    assert callback.metric_sums == {}


def test_momduck_forward_head_posture_and_stride_action_rate():
    env = object.__new__(E.MomduckForwardEnv)
    rel = np.zeros(14, dtype=np.float32)
    rel[7] = 0.6   # head_yaw turned left
    rel[8] = -0.4  # head_roll
    env._joint_pos_rel = lambda: rel
    np.testing.assert_allclose(env._head_yaw_roll_penalty(), -1.04)
    assert env.W_HEAD_POSE == 2.0

    # At zero lifetime the original ramp starts at 0.1; a warm-started policy
    # would otherwise inherit 1.0 and be charged for making a full stride.
    env._lifetime_steps = 0
    assert env._action_rate_weight() == 0.1
    env._lifetime_steps = 100_000
    assert env._action_rate_weight() == 0.3
    assert E.MomduckRunEnv.ACTION_RATE_CAP == 0.5
    assert E.MomduckRunEnv.W_HEAD_YAW_ROLL == 0.0


def _stable_run_metrics(target=0.4, **overrides):
    metrics = {
        "target_speed": target,
        "mean_speed": target,
        "speed_std": target * 0.05,
        "mean_abs_lateral": 0.01,
        "steps": 1000,
        "seconds": 20.0,
        "completed": True,
    }
    metrics.update(overrides)
    return metrics


def test_momduck_run_recipe_and_preview_use_the_same_clean_xml():
    behavior = B.BEHAVIORS["momduck_run"]
    args = T.parse_args([
        *behavior.trainer[2:],
        "--init-from", "runs/teach-momduck_stand-example-s1",
        "--envs", "12", "--steps", "30000000",
        "--run-name", "momduck-run-overnight-v1",
    ])
    assert T.env_class(args.robot, args.task).__name__ == "MomduckRunEnv"
    assert args.freeze_obs_norm
    expected = {
        "domain_rand": False,
        "obs_noise": False,
        "actuator_force": "xml",
        "max_episode_s": 20.0,
        "action_delay": False,
        "push_robot": False,
    }
    assert T.env_kwargs_from_args(args) == expected
    assert V.trainee_env_kwargs(behavior, behavior.curriculum[0].env) == {
        "task": "run", **expected}


def test_momduck_run_target_changes_only_as_a_pending_next_episode_target():
    env = object.__new__(E.MomduckRunEnv)
    env.run_target_speed = 0.4
    env._run_pending_speed = None
    env.twist_cmd = np.zeros(3)
    env.head_cmd = np.zeros(4)
    env.body_cmd = np.zeros(6)
    env.set_run_target_speed(0.6)
    assert env.run_target_speed == 0.4
    assert env._run_pending_speed == 0.6
    env._sample_commands()
    np.testing.assert_allclose(env.twist_cmd, (0.4, 0.0, 0.0))


def test_momduck_run_curriculum_requires_twelve_consecutive_stable_episodes(tmp_path):
    class FakeVec:
        def __init__(self):
            self.calls = []

        def env_method(self, name, value):
            self.calls.append((name, value))

    class FakeModel:
        def __init__(self):
            self.vec = FakeVec()
            self.num_timesteps = 12345

        def get_vec_normalize_env(self):
            return self.vec

    callback = MomduckRunCurriculum(tmp_path)
    model = FakeModel()
    callback.model = model

    for _ in range(11):
        callback._record_episode(_stable_run_metrics())
    assert model.vec.calls == []

    # One fall breaks the streak; eleven more successes are not enough.
    callback._record_episode(_stable_run_metrics(completed=False))
    for _ in range(11):
        callback._record_episode(_stable_run_metrics())
    assert model.vec.calls == []

    callback._record_episode(_stable_run_metrics())
    assert model.vec.calls == [("set_run_target_speed", 0.6)]
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["confirmed_speed"] == 0.4
    assert status["max_confirmed_speed"] == 0.4
    assert status["target_speed"] == 0.6
    assert status["promotions"] == 1

    # An episode finishing at the old target cannot confirm the new level.
    for _ in range(12):
        callback._record_episode(_stable_run_metrics(target=0.4))
    assert model.vec.calls == [("set_run_target_speed", 0.6)]


def test_momduck_trainee_previews_the_same_clean_xml_stand():
    behavior = B.BEHAVIORS["momduck_stand"]
    kwargs = V.trainee_env_kwargs(behavior, behavior.curriculum[0].env)
    assert kwargs == {
        "task": "stand",
        "max_episode_s": 20.0,
        "domain_rand": False,
        "obs_noise": False,
        "actuator_force": "xml",
        "action_delay": False,
        "push_robot": False,
    }


def test_momduck_stand_charges_the_jaw_propped_survival_pose():
    env = object.__new__(E.MomduckStandEnv)
    env.floor_geom = 0
    env._foot_body_ids = {10}
    env.model = SimpleNamespace(geom_bodyid=np.array([0, 10, 20]))
    env.data = SimpleNamespace(
        ncon=2,
        contact=SimpleNamespace(
            geom1=np.array([0, 0]), geom2=np.array([1, 2])),
    )
    assert env._non_foot_contact_penalty() == -env.W_NON_FOOT_CONTACT

    env.data.contact.geom2 = np.array([1, 1])
    assert env._non_foot_contact_penalty() == 0.0
    np.testing.assert_allclose(
        E.MomduckStandEnv.STAND_FALL_GRAVITY_Z, -np.cos(np.radians(45.0)))


def test_momduck_stand_keyframe_holds_open_loop():
    env = E.MomduckStandEnv(
        actuator_force="xml", max_episode_s=4.0, obs_noise=False,
        domain_rand=False, action_delay=False, push_robot=False)
    assert env.default_pose[2] < 0.0 < env.default_pose[11]
    _, _ = env.reset(seed=101)
    heights, tilts, planted = [], [], []
    for _ in range(200):
        _, _, terminated, _, _ = env.step(np.zeros(14, dtype=np.float32))
        assert not terminated
        heights.append(float(env._trunk_xpos[2]))
        gravity = env._projected_gravity()
        tilts.append(float(np.degrees(np.arctan2(
            np.linalg.norm(gravity[:2]), -gravity[2]))))
        contacts = env._foot_contacts()
        planted.append(float(contacts["left"] and contacts["right"]))
    assert abs(np.mean(heights[-50:]) - env.stand_z) < 0.005
    assert np.mean(tilts[-50:]) < 1.0
    assert np.mean(planted) > 0.99
    env.close()


def test_momduck_stand_recipe_shows_the_non_foot_contact_charge():
    behavior = B.BEHAVIORS["momduck_stand"]
    terms = {term.key: term for term in behavior.terms}
    assert terms["non_foot_contact_penalty"].is_penalty
    assert terms["non_foot_contact_penalty"].weight == 4.0


def test_momduck_teach_job_launches_train_walk_with_the_expected_flags(fake_popen):  # noqa: F811
    job = V.TrainingJob("momduck_stand", envs=12, steps=1000, snap_steps=250)
    argv = fake_popen[-1].cmd
    assert argv[1:3] == ["-m", "microduck_local.train"]
    assert argv[argv.index("--robot") + 1] == "momduck"
    assert argv[argv.index("--task") + 1] == "stand"
    assert argv[argv.index("--actuator") + 1] == "xml"
    assert argv[argv.index("--lr") + 1] == "0.0003"
    assert argv[argv.index("--envs") + 1] == "12"
    assert argv[argv.index("--steps") + 1] == "1000"
    assert argv[argv.index("--snap-steps") + 1] == "250"
    assert fake_popen[-1].kwargs["env"]["MICRODUCK_ACTUATOR"] == "xml"
    assert job.behavior.robot == "momduck"


def test_the_walk_env_cuts_a_sprawl_that_used_to_pay_for_lying():
    """Measured 2026-11-21 on teach-momduck_walk-1eb705: the policy lay at
    57-65 deg tilt, trunk_z 0.107, farming ~4.2/step for full 20 s episodes —
    the duck-inherited 70 deg / 0.07 m cuts never fired. The walk task's own
    thresholds must end that state in a step or two, and leave a stand alone."""
    env = E.MomduckWalkEnv(actuator_force="xml", seed=7, max_episode_s=20.0)
    assert env._fall_height == E.MomduckWalkEnv.WALK_FALL_HEIGHT == 0.12
    assert env._fall_gravity_z == E.MomduckWalkEnv.WALK_FALL_GRAVITY_Z == -0.65

    env.reset(seed=7)
    # The measured sprawl: 60 deg pitch, trunk at 0.107 m.
    angle = np.deg2rad(60)
    env.data.qpos[2] = 0.107
    env.data.qpos[3:7] = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    env.data.qvel[:] = 0.0
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    terminated = False
    for _ in range(5):
        _, _, terminated, _, _ = env.step(np.zeros(14, dtype=np.float32))
        if terminated:
            break
    assert terminated, "a 60 deg, 0.107 m sprawl must terminate immediately"

    env2 = E.MomduckWalkEnv(actuator_force="xml", seed=7, max_episode_s=20.0)
    env2.reset(seed=7)
    for _ in range(200):
        _, _, terminated, _, _ = env2.step(np.zeros(14, dtype=np.float32))
        assert not terminated, "zero action standing must NOT be cut"


def test_the_walk_env_charges_head_tilt_and_narrow_stance():
    """Gait-quality terms measured on 3ec736 (2026-11-21): head tilt ~25 deg
    mean (STAND ~4), foot separation 0.065 m mean (STAND 0.117, min 0.045).
    The charges must fire on those states and stay silent on a clean stand.
    Zero action is the floor: it must NOT be charged."""
    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(200):
        env.step(np.zeros(14, dtype=np.float32))
    reward, terms = env._compute_reward()
    # These two terms are stateless reads of the current data, so a direct
    # call is exact here (unlike the monotone terms in AGENTS.md's lesson).
    assert terms["head_tilt_penalty"] == 0.0, "a clean stand must not be charged"
    assert terms["foot_separation_penalty"] == 0.0

    # Planted narrow stance: drag both foot geoms to 3 cm apart in-place.
    fl = env.foot_geoms["left"]
    fr = env.foot_geoms["right"]
    mid = (env.data.geom_xpos[fl] + env.data.geom_xpos[fr]) / 2
    env.data.geom_xpos[fl][:2] = mid[:2] + np.array([0.015, 0.0])
    env.data.geom_xpos[fr][:2] = mid[:2] - np.array([0.015, 0.0])
    _, terms = env._compute_reward()
    assert terms["foot_separation_penalty"] < -0.1, "soles 3 cm apart must be charged"

    # Planted head tilt: rotate the head geom's frame 30 deg off vertical.
    import math
    a = math.radians(30)
    env.data.geom_xmat[env._head_geom] = [
        1.0, 0.0, 0.0,
        0.0, math.cos(a), -math.sin(a),
        0.0, math.sin(a), math.cos(a)]
    _, terms = env._compute_reward()
    assert terms["head_tilt_penalty"] < -1.0, "a 30 deg head tilt must be charged"


def test_retreat_pays_negative_and_earns_no_air_time():
    """Jonathan caught 54f0d7 stamping backward on 2026-11-21: feet_air_time
    paid stepping in ANY direction, so retreat kept most of the economy. The
    directional terms must price it: signed progress along the command, and
    no air-time payout while retreating."""
    import mujoco

    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(50):
        env.step(np.zeros(14, dtype=np.float32))
    # Planted retreat: command forward, body-frame forward speed -0.1 m/s.
    # The spawn yaw is randomized (commands are body-frame), so rotate the
    # body-frame velocity into the world frame through the trunk's rotation.
    env.twist_cmd[:] = (0.15, 0.0, 0.0)
    rmat = env._trunk_xmat.reshape(3, 3)
    env.data.qvel[0:3] = rmat @ np.array([-0.1, 0.0, 0.0])
    mujoco.mj_forward(env.model, env.data)
    _, terms = env._compute_reward()
    assert terms["forward_progress"] < -1.0, "retreat must pay negative"
    # And a forward slide pays positive.
    env.data.qvel[0:3] = rmat @ np.array([0.1, 0.0, 0.0])
    mujoco.mj_forward(env.model, env.data)
    _, terms = env._compute_reward()
    assert terms["forward_progress"] > 1.0, "forward progress must pay positive"


def test_stance_symmetry_is_charged_in_double_support():
    """Jonathan: "стойка не симметричная". The charge fires on left-right
    skew from the MIRRORED pose while both feet are down (single support is
    gated off in the env — the swing leg must differ)."""
    import mujoco

    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(200):
        env.step(np.zeros(14, dtype=np.float32))
    _, terms = env._compute_reward()
    assert abs(terms["stance_symmetry_penalty"]) < 1e-6, "clean stand is symmetric"

    # Planted skew: bend the left knee 20 deg off default. The knee bend
    # lifts the foot (single support), so pin the contact gate to prove the
    # charge itself — the gate has its own one-line condition above.
    adr = int(env.model.joint("left_knee").qposadr[0])
    env.data.qpos[adr] += np.deg2rad(20.0)
    mujoco.mj_forward(env.model, env.data)
    env._foot_contacts = lambda: {"left": True, "right": True}
    _, terms = env._compute_reward()
    assert terms["stance_symmetry_penalty"] < -0.01, "a 20 deg knee skew must be charged"


def test_swing_phase_pays_stride_and_clearance():
    """Jonathan (2026-11-22): motivate long strides and high leg lifts.
    Both terms pay only in single support, relative to the planted foot,
    along the body's forward axis — and stay silent in double support."""
    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(200):
        env.step(np.zeros(14, dtype=np.float32))
    _, terms = env._compute_reward()
    assert terms["stride_length"] == 0.0
    assert terms["foot_clearance"] == 0.0

    # Planted single support: right foot 8 cm along the BODY-forward axis
    # AHEAD OF THE TRUNK (v5 measures stride against the trunk projection;
    # spawn yaw is randomised) and lifted 5 cm over the planted sole.
    env._foot_contacts = lambda: {"left": True, "right": False}
    pst = env.data.geom_xpos[env.foot_geoms["left"]].copy()
    f = env._trunk_xmat.reshape(3, 3) @ env.BODY_FORWARD
    trunk = env._trunk_xpos[:2].copy()
    env.data.geom_xpos[env.foot_geoms["right"]][:2] = trunk + 0.08 * f[:2]
    env.data.geom_xpos[env.foot_geoms["right"]][2] = pst[2] + 0.05
    _, terms = env._compute_reward()
    assert terms["stride_length"] > 1.0, "an 8 cm lead must pay"
    assert terms["foot_clearance"] > 1.0, "a 5 cm lift must pay"


def test_hip_roll_symmetry_has_its_own_charge():
    """Dedicated hip_roll symmetry charge (Jonathan: "нужна симметрия
    hip roll") on top of the generic stance term — same double-support
    gate, mirror sign -1 (symmetric stance: rel_left == -rel_right)."""
    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(200):
        env.step(np.zeros(14, dtype=np.float32))
    _, terms = env._compute_reward()
    assert abs(terms["hip_roll_symmetry_penalty"]) < 1e-5
    il, _, _ = env._hip_roll_pair
    adr = int(env.model.joint("left_hip_roll").qposadr[0])
    env.data.qpos[adr] += np.deg2rad(8.0)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env._foot_contacts = lambda: {"left": True, "right": True}
    _, terms = env._compute_reward()
    assert terms["hip_roll_symmetry_penalty"] < -0.005, "an 8 deg hip_roll skew must be charged"


def test_hip_yaw_symmetry_has_its_own_charge():
    """Dedicated hip_yaw symmetry charge (Jonathan: "есть явная асимметрия
    hip yaw") — measured on e75952 in double support: both sides ~-9 deg
    where symmetry needs rel_left == -rel_right, i.e. an 18 deg skew.
    Same double-support gate, mirror sign -1, as the hip_roll charge."""
    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(200):
        env.step(np.zeros(14, dtype=np.float32))
    _, terms = env._compute_reward()
    assert abs(terms["hip_yaw_symmetry_penalty"]) < 1e-5
    iy, _, _ = env._hip_yaw_pair
    adr = int(env.model.joint("left_hip_yaw").qposadr[0])
    # A pure yaw deflection of ONE side by 9 deg; symmetric counterpart
    # would be +9 deg on the other side, so this is an 18 deg skew.
    env.data.qpos[adr] += np.deg2rad(9.0)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env._foot_contacts = lambda: {"left": True, "right": True}
    _, terms = env._compute_reward()
    assert terms["hip_yaw_symmetry_penalty"] < -0.005, "a one-sided 9 deg hip_yaw deflection must be charged"


def test_path_drift_is_charged_under_forward_command():
    """Path-straightness charge (Jonathan: "проблема с прямолинейностью" at
    0.25 m/s — measured heading RMSE ~110 deg, lateral RMS 0.144 m/s on
    e75952@15394k). Quadratic on body-frame lateral velocity and yaw-rate
    error; zero without a motion command."""
    env = E.MomduckWalkEnv(actuator_force="xml", seed=5, max_episode_s=20.0)
    env.reset(seed=5)
    for _ in range(200):
        env.step(np.zeros(14, dtype=np.float32))
    env.twist_cmd[:] = (0.25, 0.0, 0.0)
    _, terms = env._compute_reward()
    assert abs(terms["path_drift_penalty"]) < 1e-3
    env.body_lin_vel = lambda: np.array([0.25, 0.1, 0.0])
    _, terms = env._compute_reward()
    assert terms["path_drift_penalty"] < -0.04, "a 0.1 m/s sideways creep must be charged"
    env.twist_cmd[:] = 0.0
    env.body_lin_vel = lambda: np.array([0.0, 0.1, 0.0])
    _, terms = env._compute_reward()
    assert terms["path_drift_penalty"] == 0.0, "a stand command must not be taxed"


