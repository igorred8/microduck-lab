"""Momduck walking and standing environments with STS3215 actuator support."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..walk_env import MicroduckWalkEnv
from .momduck import BODY
from .momduck_bam import BamSTS3215Actuator


class MomduckWalkEnv(MicroduckWalkEnv):
    """The compatible 61-obs walker, with Momduck's own servo physics."""

    BAM_ACTUATOR_CLS = BamSTS3215Actuator
    BODY_FORWARD = (1.0, 0.0, 0.0)  # теперь X-вперёд

    # Fall thresholds calibrated to THIS body, not inherited from the duck:
    # measured on teach-momduck_walk-1eb705 (2026-11-21), the policy lay
    # sprawled at 57-65 deg tilt and trunk_z 0.107 for whole 20 s episodes,
    # farming ~4.2/step — the duck's 70 deg / 0.07 m cuts never fired, so the
    # lie cost nothing. Lying must END the episode, not pay for 20 seconds.
    WALK_FALL_HEIGHT = 0.12        # m; a sprawled trunk sits at ~0.107
    WALK_FALL_GRAVITY_Z = -0.65    # ~50 deg tilt; the lie measured 57-65 deg

    # Gait-quality calibration for THIS body (2026-11-21, measured on
    # teach-momduck_walk-3ec736 vs its STAND keyframe):
    #   track Gaussian too wide — standing earned ~80% of track (TRACK_STD2
    #     0.1, err 0.15 -> exp(-0.225)=0.80), so speed never paid;
    #   feet_air_time window [0.125, 0.3] s never opened — mean lift 2.7 mm;
    #   head roll mean 15 deg (STAND 4.4) — lateral lean nobody priced;
    #   foot separation mean 0.065 m (STAND 0.117, min 0.045 — soles touch).
    # Overrides apply to the momduck subclass only; the duck path is untouched.
    W_TRACK_LIN = 4.0
    TRACK_STD2 = 0.02
    W_AIR_TIME = 5.0
    AIR_TIME_MIN = 0.06
    # v3 (2026-11-21, after Jonathan caught two exploits live on 54f0d7):
    # the policy answered the head-tilt charge by leaning the TRUNK 11-13 deg
    # (upright's std2 0.05 still paid ~42% at that tilt), and stamped BACKWARD
    # (feet_air_time pays stepping in ANY direction, so retreating kept most
    # of the economy). Narrow upright so a 12 deg lean stops paying, and price
    # retreat directly.
    UPRIGHT_STD2 = 0.02
    W_FORWARD_PROGRESS = 2.0
    #: Stance-symmetry charge (Jonathan, watching 0fea9a: "стойка не
    #: симметричная"): squared left-right deviation from the MIRRORED pose,
    #: charged ONLY in double support — during a step the legs must differ,
    #: and charging the swing phase would tax the gait itself. Measured on
    #: 0fea9a: ankle ~29 deg, hip_pitch ~19 deg of left-right skew.
    W_STANCE_SYMMETRY = 2.0

    #: Head-tilt charge: quadratic per degree beyond a 6 deg allowance.
    #: Total tilt of the head's z-axis from world up — covers the roll Jonathan
    #: flagged (20 deg mean) and the equally off pitch (15 deg mean), without
    #: an axis-index trap (measured via the z-axis z-component, not xmat[5]).
    W_HEAD_TILT_PENALTY = 0.02
    HEAD_TILT_TOL_DEG = 6.0
    #: Stance-width charge: quadratic per metre inside the 0.08 m band
    #: (~68% of the measured STAND separation 0.117 m).
    W_FOOT_SEPARATION = 2000.0
    FOOT_SEPARATION_MIN = 0.08

    def __init__(self, *args, **kwargs):
        # Generic entry points instantiate BODY.env_class() without a `robot`
        # argument. Never let the base class silently select C.MICRODUCK.
        # Forward/stand/run set their own `robot` kwargs before this call, so
        # the calibrated thresholds apply to the plain walk task only.
        kwargs.setdefault("robot", replace(
            BODY, fall_height=self.WALK_FALL_HEIGHT,
            fall_gravity_z=self.WALK_FALL_GRAVITY_Z))
        super().__init__(*args, **kwargs)
        import mujoco
        self._head_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_soft")
        # Left leg (action indices 0-4) and its mirror partners: perm maps
        # 0..4 -> 9..13 (right leg). neck/head (5-8) are excluded — the head
        # has its own tilt charge.
        from .momduck_symmetry import momduck_mirror_action
        _p = np.asarray(momduck_mirror_action.perm)
        _s = np.asarray(momduck_mirror_action.sign)
        self._sym_pairs = [(i, int(_p[i]), float(_s[i])) for i in range(5)]

    def _compute_reward(self):
        reward, terms = super()._compute_reward()
        # Head tilt: angle of the head geom's z-axis away from world up.
        m = self.data.geom_xmat[self._head_geom]
        tilt_deg = float(np.degrees(np.arccos(np.clip(m[8], 0.0, 1.0))))
        over = max(0.0, tilt_deg - self.HEAD_TILT_TOL_DEG)
        terms["head_tilt_penalty"] = -self.W_HEAD_TILT_PENALTY * over * over
        # Stance width: horizontal distance between the foot geoms.
        fl = self.data.geom_xpos[self.foot_geoms["left"]]
        fr = self.data.geom_xpos[self.foot_geoms["right"]]
        sep = float(np.linalg.norm(fl[:2] - fr[:2]))
        narrow = max(0.0, self.FOOT_SEPARATION_MIN - sep)
        terms["foot_separation_penalty"] = -self.W_FOOT_SEPARATION * narrow * narrow
        reward += terms["head_tilt_penalty"] + terms["foot_separation_penalty"]
        # Directional terms: signed progress along the commanded body-frame
        # direction, and NO air-time payout while retreating from it. Without
        # the second half, backward stamping kept ~8/step of the economy.
        cmd = self.twist_cmd
        cmd_speed = float(np.hypot(float(cmd[0]), float(cmd[1])))
        if cmd_speed > 0.01:
            v = self.body_lin_vel()
            proj = float(v[0] * cmd[0] + v[1] * cmd[1]) / cmd_speed
            terms["forward_progress"] = self.W_FORWARD_PROGRESS * float(
                np.clip(proj / cmd_speed, -1.0, 1.0))
            if proj < 0.0 and terms["feet_air_time"] > 0.0:
                reward -= terms["feet_air_time"]
                terms["feet_air_time"] = 0.0
        else:
            terms["forward_progress"] = 0.0
        reward += terms["forward_progress"]
        # Stance symmetry: left-right skew from the mirrored pose, charged
        # only when BOTH feet are down — the swing phase must stay free.
        contacts = self._foot_contacts()
        if contacts["left"] and contacts["right"]:
            rel = self._joint_pos_rel()
            skew = 0.0
            for i, j, s in self._sym_pairs:
                d = float(rel[i]) - s * float(rel[j])
                skew += d * d
            terms["stance_symmetry_penalty"] = -self.W_STANCE_SYMMETRY * skew
        else:
            terms["stance_symmetry_penalty"] = 0.0
        reward += terms["stance_symmetry_penalty"]
        return reward, terms



    def heading_lin_vel(self) -> tuple[float, float, float]:
        """Momduck equivalent of the base yaw-heading velocity readout."""
        v = self._qvel_base
        forward = self._trunk_xmat.reshape(3, 3) @ self.BODY_FORWARD
        forward[2] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm > 1e-9:
            forward /= norm
        else:
            forward = np.array(self.BODY_FORWARD, dtype=float)
        side = np.array([-forward[1], forward[0], 0.0])
        return (float(v @ forward), float(v @ side), float(v[2]))


class MomduckForwardEnv(MomduckWalkEnv):
    """Walk straight forward from the verified stance, without crouching.

    This is the first locomotion curriculum, not the general twist-command
    walker: the abstract forward command is always +0.15 m/s, lateral and yaw
    commands are zero, and the head/body auxiliary commands are calm. Momduck's
    corrected MJCF faces body +X, so all physical tracking and heading
    measurements use that axis. The policy's command slots stay unchanged.
    A nominal-height reward and a 0.14 m fall threshold reject the crouched
    survival strategy found by the original walking run. Velocity and heading
    are measured against the episode's initial course, with explicit lateral
    and heading penalties, so a sideways shuffle cannot pass as forward gait.
    Loaded legs and soles are additionally anchored to STAND without constraining
    the swing leg; MomduckRunEnv disables this walking-specific posture prior.
    """

    FORWARD_SPEED = 0.15  # m/s — deliberately below the old 0.3 m/s minimum
    FORWARD_FALL_HEIGHT = 0.14  # nominal STAND is about 0.156 m

    W_TRACK_LIN = 4.0
    TRACK_STD2 = 0.02  # standing still earns little velocity-tracking reward
    W_TRACK_ANG = 1.0
    W_HEAD_POSE = 2.0  # original Microduck walk weight (was incorrectly weakened)
    W_HEAD_YAW_ROLL = 4.0
    _HEAD_YAW_ROLL_IDS = np.array([7, 8])
    W_POSE = 0.2  # do not tax the sagittal joint excursion needed for a stride
    W_AIR_TIME = 3.0  # original Microduck walk weight
    W_PROGRESS = 2.0
    W_HEIGHT = 3.0
    HEIGHT_STD2 = 0.004
    W_LATERAL = 20.0
    # 1-cos(yaw) is almost free at small angles: at 10 deg the old weight 2
    # charged only 0.03/step, so a steady left turn beat the posture rewards.
    # Price the actual heading angle and yaw rate with useful small-error gradients.
    W_HEADING = 12.0
    W_YAW_RATE = 12.0
    # Air-time alone has a 0.125 s dead zone: a shuffling policy gets no signal
    # until it already dares to lift a foot. Pay measured clearance densely, but
    # only during SINGLE support so a two-foot hop cannot collect the reward.
    W_FOOT_CLEARANCE = 4.0
    FOOT_CLEARANCE_TARGET = 0.03  # original Microduck run's 3 cm swing target
    # Clearance alone rewards lifting, not striding. During single support pay a
    # dense forward foot-separation gradient from trailing to leading swing foot.
    W_SWING_STRIDE = 3.0
    FOOT_STRIDE_TARGET = 0.08  # 8 cm from 4 cm trailing to 4 cm leading
    ACTION_RATE_CAP = 0.3  # warm starts otherwise inherit the full 1.0 stride tax

    # Walking-only support posture. A quadratic penalty retains a useful gradient
    # after the base exponential pose reward has saturated near zero. Yaw/roll
    # deviations splay or twist the support leg and are therefore priced more
    # strongly than the sagittal joints that must move to make a stride.
    W_STANCE_POSE = 2.0
    W_STANCE_FLAT = 1.0
    W_STANCE_YAW = 2.0
    # MJCF hip_roll range is only [-25, +20] deg, while STAND uses -/+5 deg.
    # The v4 policy parked near those limits because the support-only average
    # was cheaper than forward progress. Roll is not a required forward-stride
    # DOF, so centre BOTH legs throughout the step and add a soft-limit barrier.
    W_HIP_ROLL = 20.0
    W_HIP_ROLL_LIMIT = 40.0
    HIP_ROLL_FREE = 0.15  # rad around DEFAULT_POSE for ordinary lateral balance
    _HIP_ROLL_IDS = np.array([1, 10])
    # v12 showed that simply increasing this cost to 24/48 breaks the donor's
    # balance around the end of the ramp. The v13 diagnostic continuation goes
    # back to v11's stable 12/24 objective, fully active from step zero, while
    # recording the actual height and hip angles. Do not raise speed or posture
    # weights until those measurements show what the balancing gait is doing.
    W_HIP_YAW = 12.0
    W_HIP_YAW_LIMIT = 24.0
    HIP_YAW_FREE_START = 0.15
    HIP_YAW_FREE = 0.15  # rad: v11's stable soft-limit allowance
    HIP_YAW_RAMP_STEPS = 60_000  # inactive while START_SCALE is 1.0
    HIP_YAW_START_SCALE = 1.0  # no new posture ramp in the diagnostic run
    _HIP_YAW_IDS = np.array([0, 9])
    # Target excess is measured below for diagnostics only. v15 charged it in
    # the environment reward and destroyed the donor's balance; v16 restores
    # the exact v13 reward and applies a separate hinge to the actor mean.
    # The failed v4 showed that a mild all-joint anchor buys speed, not posture.
    # Concentrate on hip yaw/roll (splay and toe-in); sagittal joints need room
    # to stride, so pitch/knee get only a small residual and the ankle is handled
    # by the measured sole-orientation terms below.
    STANCE_POSE_WEIGHTS = np.array([1.0, 1.0, 0.05, 0.05, 0.0])
    STANCE_FLAT_STD2 = 0.45 ** 2
    _LEFT_LEG_IDS = np.arange(0, 5)
    _RIGHT_LEG_IDS = np.arange(9, 14)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot", replace(BODY, fall_height=self.FORWARD_FALL_HEIGHT))
        kwargs.setdefault("max_episode_s", 20.0)
        kwargs.setdefault("obs_noise", False)
        kwargs.setdefault("domain_rand", False)
        kwargs.setdefault("action_delay", False)
        kwargs.setdefault("push_robot", False)
        # The first gait is learned along one viewer/world axis. Random yaw can
        # return after the nominal straight gait exists.
        kwargs.setdefault("random_yaw", False)
        super().__init__(*args, **kwargs)
        # Read the limits from the verified MJCF; do not duplicate or alter its
        # geometry/joint specification. Convert absolute q targets to the same
        # DEFAULT_POSE-relative convention as the policy action and diagnostics.
        yaw_joint_ids = np.array([
            self.model.joint(self.robot.joint_names[i]).id
            for i in self._HIP_YAW_IDS
        ])
        self._hip_yaw_target_lo = (
            self.model.jnt_range[yaw_joint_ids, 0]
            - self.default_pose[self._HIP_YAW_IDS]
        )
        self._hip_yaw_target_hi = (
            self.model.jnt_range[yaw_joint_ids, 1]
            - self.default_pose[self._HIP_YAW_IDS]
        )
        self._spawn_forward = np.array(self.BODY_FORWARD, dtype=float)
        self._spawn_side = np.array([
            -self._spawn_forward[1], self._spawn_forward[0], 0.0])
        # super().__init__ leaves mjData at STAND. Each foot's current local
        # gravity is therefore its measured nominal-flat reference; no MJCF
        # axis convention or visual geometry is assumed or changed.
        self.foot_flat_ref = {}
        self.foot_yaw_axes = {}
        self.foot_forward_ref = {}
        self.foot_stand_z = {}
        body_forward = np.array(self.BODY_FORWARD, dtype=float)
        for side, gid in self.foot_geoms.items():
            self.foot_stand_z[side] = float(self.data.geom_xpos[gid][2])
            rot = self.data.geom_xmat[gid].reshape(3, 3)
            self.foot_flat_ref[side] = rot.T @ np.array([0.0, 0.0, -1.0])
            # Discover the foot-local axis that points forward at STAND instead
            # of assuming a geom-frame convention. Sole flatness is invariant to
            # vertical rotation, so it alone cannot reject the v4 toe-in gait.
            local_forward = rot.T @ body_forward
            axis = int(np.argmax(np.abs(local_forward)))
            sign = 1.0 if local_forward[axis] >= 0.0 else -1.0
            reference = sign * rot[:, axis]
            reference[2] = 0.0
            reference /= max(float(np.linalg.norm(reference)), 1e-9)
            self.foot_yaw_axes[side] = (axis, sign)
            self.foot_forward_ref[side] = reference

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        forward = self._trunk_xmat.reshape(3, 3) @ self.BODY_FORWARD
        forward[2] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm > 1e-9:
            forward /= norm
        else:
            forward = np.array(self.BODY_FORWARD, dtype=float)
        self._spawn_forward = forward
        self._spawn_side = np.array([-forward[1], forward[0], 0.0])
        return obs, info

    def _target_speed(self) -> float:
        return float(self.FORWARD_SPEED)

    def _action_rate_weight(self) -> float:
        # The base ramp reaches 1.0 almost immediately on a warm-started run.
        # Cap it while learning a full stride; RUN gets the original run's 0.5.
        return min(super()._action_rate_weight(), self.ACTION_RATE_CAP)

    def _sample_commands(self) -> None:
        self.twist_cmd[:] = (self._target_speed(), 0.0, 0.0)
        self.head_cmd[:] = 0.0
        self.body_cmd[:] = 0.0

    def _walking_stance_terms(self) -> tuple[float, float, float, float]:
        """Anchor only loaded feet/legs; leave the swing leg free to step.

        Unlike another Gaussian reward, these bounded-cost quadratic/one-minus-
        Gaussian terms still point back toward STAND from a badly splayed gait.
        Foot yaw is separate from flatness: a level sole can still point inward.
        RUN disables all three weights so flight and dynamic support stay free.
        """
        if (self.W_STANCE_POSE == 0.0 and self.W_STANCE_FLAT == 0.0
                and self.W_STANCE_YAW == 0.0 and self.W_FOOT_CLEARANCE == 0.0):
            return 0.0, 0.0, 0.0, 0.0
        contacts = self._foot_contacts()
        rel = self._joint_pos_rel()
        pose_cost = 0.0
        pose_norm = 0.0
        flat_cost = 0.0
        yaw_cost = 0.0
        clearance_reward = 0.0
        n_stance = 0
        for side, other, ids in (
                ("left", "right", self._LEFT_LEG_IDS),
                ("right", "left", self._RIGHT_LEG_IDS)):
            if not contacts[side] and contacts[other]:
                gid = self.foot_geoms[side]
                lift = (float(self.data.geom_xpos[gid][2])
                        - self.foot_stand_z[side])
                clearance_reward += float(np.clip(
                    lift / self.FOOT_CLEARANCE_TARGET, 0.0, 1.0))
            if not contacts[side]:
                continue
            n_stance += 1
            d2 = rel[ids] ** 2
            pose_cost += float(d2 @ self.STANCE_POSE_WEIGHTS)
            pose_norm += float(self.STANCE_POSE_WEIGHTS.sum())
            gid = self.foot_geoms[side]
            rot = self.data.geom_xmat[gid].reshape(3, 3)
            g_local = -self.data.geom_xmat[gid][6:9]
            delta2 = float(((g_local - self.foot_flat_ref[side]) ** 2).sum())
            flat_cost += 1.0 - float(np.exp(-delta2 / self.STANCE_FLAT_STD2))

            axis, sign = self.foot_yaw_axes[side]
            forward = sign * rot[:, axis]
            forward[2] = 0.0
            norm = float(np.linalg.norm(forward))
            if norm > 1e-9:
                forward /= norm
                yaw_cos = float(np.clip(
                    forward @ self.foot_forward_ref[side], -1.0, 1.0))
                yaw_cost += 1.0 - yaw_cos
            else:
                # A vertical foot has no meaningful forward heading; charge a
                # right-angle error rather than letting a norm singularity pay.
                yaw_cost += 1.0
        if not n_stance:
            return 0.0, 0.0, 0.0, 0.0
        pose_penalty = -self.W_STANCE_POSE * pose_cost / pose_norm
        flat_penalty = -self.W_STANCE_FLAT * flat_cost / n_stance
        yaw_penalty = -self.W_STANCE_YAW * yaw_cost / n_stance
        clearance = self.W_FOOT_CLEARANCE * clearance_reward
        return (float(pose_penalty), float(flat_penalty),
                float(yaw_penalty), float(clearance))

    def _swing_stride_reward(self) -> float:
        """Pay forward swing-foot travel, not merely foot height or air time.

        With exactly one loaded foot, measure the other's signed longitudinal
        separation from the support sole along the episode's initial course.
        The dense ramp is zero at 4 cm trailing, half at the support foot and
        saturated at 4 cm leading. Both feet down or a two-foot hop pay nothing;
        the caller additionally gates this by achieved forward velocity.
        """
        if self.W_SWING_STRIDE == 0.0:
            return 0.0
        contacts = self._foot_contacts()
        if contacts["left"] == contacts["right"]:
            return 0.0
        swing = "right" if contacts["left"] else "left"
        stance = "left" if contacts["left"] else "right"
        delta = (self.data.geom_xpos[self.foot_geoms[swing]]
                 - self.data.geom_xpos[self.foot_geoms[stance]])
        along = float(delta @ self._spawn_forward)
        target = self.FOOT_STRIDE_TARGET
        phase = float(np.clip((along + 0.5 * target) / target, 0.0, 1.0))
        return self.W_SWING_STRIDE * phase

    def _head_yaw_roll_penalty(self) -> float:
        """Keep a turned/rolled head inside the Gaussian's useful gradient."""
        if self.W_HEAD_YAW_ROLL == 0.0:
            return 0.0
        rel = self._joint_pos_rel()[self._HEAD_YAW_ROLL_IDS].astype(float)
        return -self.W_HEAD_YAW_ROLL * float(np.mean(rel ** 2))

    def _hip_roll_terms(self) -> tuple[float, float]:
        """Keep both hips off their narrow roll limits, including in swing."""
        if self.W_HIP_ROLL == 0.0 and self.W_HIP_ROLL_LIMIT == 0.0:
            return 0.0, 0.0
        roll = self._joint_pos_rel()[self._HIP_ROLL_IDS].astype(float)
        centre_penalty = -self.W_HIP_ROLL * float(np.mean(roll ** 2))
        excess = np.maximum(np.abs(roll) - self.HIP_ROLL_FREE, 0.0)
        limit_penalty = -self.W_HIP_ROLL_LIMIT * float(np.mean(excess ** 2))
        return float(centre_penalty), float(limit_penalty)

    def _hip_yaw_weight_scale(self) -> float:
        """Return the configured posture schedule (v13 holds it at 1.0)."""
        progress = float(np.clip(
            self._lifetime_steps / self.HIP_YAW_RAMP_STEPS, 0.0, 1.0))
        start = self.HIP_YAW_START_SCALE
        return start + (1.0 - start) * progress

    def _hip_yaw_free_band(self) -> float:
        """Narrow the extra barrier's allowance without a warm-start jump.

        A later speed-only stage can set HIP_YAW_START_SCALE=1.0 to keep the
        final posture objective fully active from its first environment step.
        """
        if self.HIP_YAW_START_SCALE >= 1.0:
            return float(self.HIP_YAW_FREE)
        progress = float(np.clip(
            self._lifetime_steps / self.HIP_YAW_RAMP_STEPS, 0.0, 1.0))
        start = self.HIP_YAW_FREE_START
        return float(start + (self.HIP_YAW_FREE - start) * progress)

    def _hip_yaw_terms(self) -> tuple[float, float]:
        """Reject Chaplin-like toe-out gradually on BOTH stance and swing legs."""
        if self.W_HIP_YAW == 0.0 and self.W_HIP_YAW_LIMIT == 0.0:
            return 0.0, 0.0
        scale = self._hip_yaw_weight_scale()
        yaw = self._joint_pos_rel()[self._HIP_YAW_IDS].astype(float)
        centre_penalty = -self.W_HIP_YAW * scale * float(np.mean(yaw ** 2))
        excess = np.maximum(
            np.abs(yaw) - self._hip_yaw_free_band(), 0.0)
        limit_penalty = -self.W_HIP_YAW_LIMIT * scale * float(np.mean(excess ** 2))
        return float(centre_penalty), float(limit_penalty)

    def _hip_yaw_target_rel(self) -> np.ndarray:
        """Applied hip-yaw position targets, relative to DEFAULT_POSE."""
        ctrl = (self.data.ctrl if self._ctrl_whole
                else self.data.ctrl[self.ctrl_adr])
        target_rel = np.asarray(ctrl) - self.default_pose
        return target_rel[self._HIP_YAW_IDS].astype(float)

    def _hip_yaw_target_excess(self, target_rel=None) -> np.ndarray:
        """Non-negative target travel outside each joint's actual MJCF range."""
        if target_rel is None:
            target_rel = self._hip_yaw_target_rel()
        above = np.maximum(target_rel - self._hip_yaw_target_hi, 0.0)
        below = np.maximum(self._hip_yaw_target_lo - target_rel, 0.0)
        return above + below

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        _total, terms = super()._compute_reward()

        dz = float(self._trunk_xpos[2]) - self.stand_z
        terms["height"] = self.W_HEIGHT * float(
            np.exp(-(dz * dz) / self.HEIGHT_STD2))

        # Measure travel along the episode's INITIAL heading, not the current
        # body axis. Otherwise a 90-degree turn can rename sideways travel as
        # "forward". BODY_FORWARD maps the abstract command to Momduck's +X.
        world_v = self._qvel_base
        vx = float(world_v @ self._spawn_forward)
        vy = float(world_v @ self._spawn_side)
        vz = float(world_v[2])
        target = self._target_speed()
        # Original Microduck's postmortem shows ungated air-time can pay marching
        # in place or sideways. Keep the original weight, but make both step-
        # shaping terms amplify ACHIEVED forward travel instead of replacing it.
        forward_gate = float(np.clip(vx / target, 0.0, 1.0))
        terms["feet_air_time"] *= forward_gate
        track_err2 = ((target - vx) ** 2 + vy ** 2 + vz ** 2)
        terms["track_lin_vel"] = self.W_TRACK_LIN * float(np.exp(
            -track_err2 / self.TRACK_STD2))
        terms["forward_progress"] = self.W_PROGRESS * float(np.clip(
            vx / target, -1.0, 1.0))
        terms["lateral_velocity_penalty"] = -self.W_LATERAL * vy ** 2

        current = self._trunk_xmat.reshape(3, 3) @ self.BODY_FORWARD
        current[2] = 0.0
        norm = float(np.linalg.norm(current))
        if norm > 1e-9:
            current /= norm
        heading_cos = float(np.clip(current @ self._spawn_forward, -1.0, 1.0))
        heading_side = float(current @ self._spawn_side)
        heading_angle = float(np.arctan2(heading_side, heading_cos))
        # Cap only the gross-error cost: a fallen/backward robot must not create
        # an unbounded penalty that overwhelms recovery and termination signals.
        heading_cost = min(heading_angle ** 2, (0.5 * np.pi) ** 2)
        terms["heading_penalty"] = -self.W_HEADING * heading_cost
        terms["yaw_rate_penalty"] = -self.W_YAW_RATE * float(self._gyro[2] ** 2)
        stance_pose, stance_flat, stance_yaw, clearance = \
            self._walking_stance_terms()
        terms["stance_pose_penalty"] = stance_pose
        terms["stance_foot_penalty"] = stance_flat
        terms["stance_foot_yaw_penalty"] = stance_yaw
        terms["foot_clearance"] = clearance * forward_gate
        terms["swing_stride"] = self._swing_stride_reward() * forward_gate
        hip_roll, hip_roll_limit = self._hip_roll_terms()
        terms["hip_roll_penalty"] = hip_roll
        terms["hip_roll_limit_penalty"] = hip_roll_limit
        hip_yaw, hip_yaw_limit = self._hip_yaw_terms()
        terms["hip_yaw_penalty"] = hip_yaw
        terms["hip_yaw_limit_penalty"] = hip_yaw_limit
        terms["head_yaw_roll_penalty"] = self._head_yaw_roll_penalty()
        return float(sum(terms.values())), terms

    def _diagnostic_metrics(self) -> dict[str, float]:
        """Measured gait state for training diagnostics, never a reward input.

        Angles are relative to DEFAULT_POSE and reported in degrees so the
        progress log directly answers whether the hips are actually centring.
        Height and course velocity come from the same post-physics state as the
        reward; publishing them does not alter observations or control.
        """
        rel = self._joint_pos_rel()
        ctrl = (self.data.ctrl if self._ctrl_whole
                else self.data.ctrl[self.ctrl_adr])
        target_rel = np.asarray(ctrl) - self.default_pose
        yaw_target = target_rel[self._HIP_YAW_IDS].astype(float)
        yaw_target_excess = self._hip_yaw_target_excess(yaw_target)
        world_v = self._qvel_base
        height = float(self._trunk_xpos[2])
        return {
            "height_m": height,
            "height_rel_stand_m": height - float(self.stand_z),
            "fall_margin_m": height - float(self._fall_height),
            "forward_speed_mps": float(world_v @ self._spawn_forward),
            "lateral_speed_mps": float(world_v @ self._spawn_side),
            "hip_yaw_left_deg": float(np.degrees(rel[self._HIP_YAW_IDS[0]])),
            "hip_yaw_right_deg": float(np.degrees(rel[self._HIP_YAW_IDS[1]])),
            "hip_yaw_target_left_deg": float(
                np.degrees(target_rel[self._HIP_YAW_IDS[0]])),
            "hip_yaw_target_right_deg": float(
                np.degrees(target_rel[self._HIP_YAW_IDS[1]])),
            "hip_yaw_target_excess_left_deg": float(
                np.degrees(yaw_target_excess[0])),
            "hip_yaw_target_excess_right_deg": float(
                np.degrees(yaw_target_excess[1])),
            "hip_roll_left_deg": float(np.degrees(rel[self._HIP_ROLL_IDS[0]])), 
            "hip_roll_right_deg": float(np.degrees(rel[self._HIP_ROLL_IDS[1]])),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["momduck_metrics"] = self._diagnostic_metrics()
        return obs, reward, terminated, truncated, info


class MomduckRunEnv(MomduckForwardEnv):
    """Performance-gated forward running, starting at 0.4 m/s.

    The trainer's MomduckRunCurriculum callback synchronises target changes
    across the fleet. A pending target takes effect only on the next reset, so
    every episode's metrics describe exactly one requested speed.
    """

    START_SPEED = 0.4
    SPEED_STEP = 0.2
    MAX_SPEED = 2.0

    # Do not import the walking support-style prior into running. In particular,
    # a flight phase and a more extended/sagittal support leg are legitimate.
    W_STANCE_POSE = 0.0
    W_STANCE_FLAT = 0.0
    W_STANCE_YAW = 0.0
    W_FOOT_CLEARANCE = 0.0
    W_SWING_STRIDE = 0.0
    W_HEAD_YAW_ROLL = 0.0
    ACTION_RATE_CAP = 0.5
    W_HIP_ROLL = 0.0
    W_HIP_ROLL_LIMIT = 0.0
    W_HIP_YAW = 0.0
    W_HIP_YAW_LIMIT = 0.0

    def __init__(self, *args, run_target_speed=None, **kwargs):
        initial = self.START_SPEED if run_target_speed is None else run_target_speed
        self.run_target_speed = float(np.clip(
            initial, self.START_SPEED, self.MAX_SPEED))
        self._run_pending_speed = None
        super().__init__(*args, **kwargs)
        self._reset_run_metrics()

    def _target_speed(self) -> float:
        return float(self.run_target_speed)

    def set_run_target_speed(self, speed: float) -> None:
        """Queue a new fleet target for this env's next episode."""
        self._run_pending_speed = float(np.clip(
            speed, self.START_SPEED, self.MAX_SPEED))

    def _reset_run_metrics(self) -> None:
        self._run_steps = 0
        self._run_speed_sum = 0.0
        self._run_speed_m2 = 0.0
        self._run_lateral_abs_sum = 0.0

    def reset(self, *, seed=None, options=None):
        if self._run_pending_speed is not None:
            self.run_target_speed = self._run_pending_speed
            self._run_pending_speed = None
        obs, info = super().reset(seed=seed, options=options)
        self._reset_run_metrics()
        return obs, info

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        total, terms = super()._compute_reward()
        world_v = self._qvel_base
        vx = float(world_v @ self._spawn_forward)
        vy = float(world_v @ self._spawn_side)
        self._run_steps += 1
        mean_before = self._run_speed_sum / (self._run_steps - 1) \
            if self._run_steps > 1 else 0.0
        self._run_speed_sum += vx
        mean_after = self._run_speed_sum / self._run_steps
        self._run_speed_m2 += (vx - mean_before) * (vx - mean_after) \
            if self._run_steps > 1 else 0.0
        self._run_lateral_abs_sum += abs(vy)
        return total, terms

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            n = max(1, self._run_steps)
            mean = self._run_speed_sum / n
            info["momduck_run"] = {
                "target_speed": float(self.run_target_speed),
                "mean_speed": float(mean),
                "speed_std": float(np.sqrt(max(0.0, self._run_speed_m2 / n))),
                "mean_abs_lateral": float(self._run_lateral_abs_sum / n),
                "steps": int(self._run_steps),
                "seconds": float(self._run_steps * 0.02),
                "completed": bool(truncated and not terminated),
            }
        return obs, reward, terminated, truncated, info


class MomduckStandEnv(MomduckWalkEnv):
    """Hold DEFAULT_POSE upright, at the nominal height and without drifting.

    The observation/action contract is identical to the walker. Only the task
    commands and reward change. The first curriculum stage deliberately has no
    observation noise, domain randomization, action delay or velocity pushes.
    A tilt past 45 degrees ends the episode and a non-foot floor contact is
    charged, so propping the head or jaw on the floor cannot pass as standing.
    """

    STAND_FALL_HEIGHT = 0.12  # nominal STAND is about 0.156 m; 0.089 m is a crouch
    STAND_FALL_GRAVITY_Z = float(-np.cos(np.radians(45.0)))

    STILL_STD2 = 0.02
    SPIN_STD2 = 0.10
    HEIGHT_STD2 = 0.004
    POSE_STD2 = 1.0
    ANCHOR_STD2 = 0.0025

    W_STILL = 3.0
    W_SPIN = 1.0
    W_UPRIGHT = 2.0
    W_HEIGHT = 3.0
    W_PLANTED = 1.0
    W_POSE = 2.0
    W_ANCHOR = 1.0
    W_JOINT_VEL = 0.002
    W_NON_FOOT_CONTACT = 4.0
    ACTION_RATE_W = 0.1

    def __init__(self, *args, **kwargs):
        # A stand-specific termination rejects the low crouch discovered by the
        # walking run without changing the walking spec or the MJCF.
        kwargs.setdefault("robot", replace(
            BODY,
            fall_height=self.STAND_FALL_HEIGHT,
            fall_gravity_z=self.STAND_FALL_GRAVITY_Z,
        ))
        kwargs.setdefault("max_episode_s", 20.0)
        kwargs.setdefault("obs_noise", False)
        kwargs.setdefault("domain_rand", False)
        kwargs.setdefault("action_delay", False)
        kwargs.setdefault("push_robot", False)
        super().__init__(*args, **kwargs)
        self._spawn_xy = np.zeros(2)
        self._foot_body_ids = {
            int(self.model.geom_bodyid[gid]) for gid in self.foot_geoms.values()
        }

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._spawn_xy = np.asarray(self._trunk_xpos[:2], float).copy()
        return obs, info

    def _non_foot_contact_penalty(self) -> float:
        for i in range(self.data.ncon):
            g1 = int(self.data.contact.geom1[i])
            g2 = int(self.data.contact.geom2[i])
            if self.floor_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.floor_geom else g1
            if int(self.model.geom_bodyid[other]) not in self._foot_body_ids:
                return -self.W_NON_FOOT_CONTACT
        return 0.0

    def _sample_commands(self) -> None:
        # This is an idle policy: all locomotion and auxiliary command slots are
        # pinned to zero. train.py records task=stand as pinned_command=True.
        self.twist_cmd[:] = 0.0
        self.head_cmd[:] = 0.0
        self.body_cmd[:] = 0.0

    def _action_rate_weight(self) -> float:
        # Do not ramp this penalty: a standing policy otherwise learns to go
        # limp when the walking reward ramp reaches full strength.
        return self.ACTION_RATE_W

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        self._lifetime_steps += 1
        velocity = self.body_lin_vel()
        gyro = self._gyro
        gravity = self._projected_gravity()

        still = self.W_STILL * float(np.exp(
            -float((velocity ** 2).sum()) / self.STILL_STD2))
        no_spin = self.W_SPIN * float(np.exp(
            -float((gyro ** 2).sum()) / self.SPIN_STD2))

        tilt2 = float((gravity[:2] ** 2).sum())
        upright = self.W_UPRIGHT * float(np.exp(-tilt2 / self.UPRIGHT_STD2))

        dz = float(self._trunk_xpos[2]) - self.stand_z
        height = self.W_HEIGHT * float(np.exp(-(dz * dz) / self.HEIGHT_STD2))

        contacts = self._foot_contacts()
        planted = self.W_PLANTED * float(contacts["left"] and contacts["right"])

        # All 14 joints, including the head, are held near DEFAULT_POSE.
        joint_pos_rel = self._joint_pos_rel()
        pose = self.W_POSE * float(np.exp(
            -float((joint_pos_rel ** 2).sum()) / self.POSE_STD2))

        dxy = self._trunk_xpos[:2] - self._spawn_xy
        stay_home = self.W_ANCHOR * float(np.exp(
            -float((dxy ** 2).sum()) / self.ANCHOR_STD2))

        action_rate = self._action_rate_weight() * -float(
            ((self.last_action - self.prev_action) ** 2).sum())
        joint_vel = self.W_JOINT_VEL * -float((self._joint_vel() ** 2).sum())

        terms = {
            "still": still,
            "no_spin": no_spin,
            "upright": upright,
            "height": height,
            "feet_planted": planted,
            "pose": pose,
            "stay_home": stay_home,
            "non_foot_contact_penalty": self._non_foot_contact_penalty(),
            "action_rate_penalty": action_rate,
            "joint_vel_penalty": joint_vel,
        }
        return float(sum(terms.values())), terms


__all__ = ["MomduckWalkEnv", "MomduckForwardEnv", "MomduckRunEnv", "MomduckStandEnv"]
