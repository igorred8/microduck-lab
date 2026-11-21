"""NumPy/CPU BAM for Momduck's Feetech STS3215 7.4 V servos (m1)."""

from __future__ import annotations

import numpy as np

from ..bam_actuator import BamXL330Actuator
from .momduck_constants import (
    MOMDUCK_DELAY_MAX_LAG,
    MOMDUCK_DELAY_MIN_LAG,
    MOMDUCK_VIN_MIN,
    MOMDUCK_VIN_RANGE,
    STS3215_ERROR_GAIN,
    STS3215_KP_FW,
    STS3215_M1_PARAMS,
    STS3215_MAX_PWM,
)


class BamSTS3215Actuator(BamXL330Actuator):
    """Reuse the CPU BAM pipeline with the STS3215 firmware and m1 friction.

    The base class supplies motor torque, battery sag, delay-buffer plumbing,
    MuJoCo indexing and friction constraints. STS3215 differs in three places:
    its firmware rate-limits the internal position target, its PWM conversion
    uses error_gain=0.166/max_pwm=0.97, and m1 has constant Coulomb friction
    plus viscous damping rather than the XL330 m6 load-dependent budget.
    """

    def __init__(self, model, data, joint_names, *, dt, rng=None, **kwargs):
        # walk_env supplies XL330 defaults. Replace them rather than allowing a
        # shared default to silently turn this back into the wrong servo.
        for key in (
            "kp_fw", "vin_range", "vin_min", "delay_min_lag",
            "delay_max_lag", "max_current", "params",
        ):
            kwargs.pop(key, None)

        # The base constructor reads the m6 fields into hot-loop attributes.
        # They are zeroed here only for API compatibility; friction_budget()
        # below implements the actual m1 model and never uses those fields.
        params = {
            "friction_stribeck": 0.0,
            "load_friction_motor": 0.0,
            "load_friction_external": 0.0,
            "load_friction_motor_stribeck": 0.0,
            "load_friction_external_stribeck": 0.0,
            "load_friction_motor_quad": 0.0,
            "load_friction_external_quad": 0.0,
            "dtheta_stribeck": 1.0,
            "alpha": 1.0,
            **STS3215_M1_PARAMS,
        }
        super().__init__(
            model,
            data,
            joint_names,
            dt=dt,
            rng=rng,
            kp_fw=STS3215_KP_FW,
            vin_range=MOMDUCK_VIN_RANGE,
            vin_min=MOMDUCK_VIN_MIN,
            delay_min_lag=MOMDUCK_DELAY_MIN_LAG,
            delay_max_lag=MOMDUCK_DELAY_MAX_LAG,
            # The official STS3215 voltage controller has no XL330-style
            # firmware current limiter.
            max_current=None,
            params=params,
            **kwargs,
        )
        self.error_gain = STS3215_ERROR_GAIN * self.p["error_gain_ratio"]
        self.max_pwm = STS3215_MAX_PWM
        self.max_velocity = float(self.p["max_velocity"])
        self.q_target_smooth: np.ndarray | None = None

        # The fused XL330 path bypasses the two overridden methods below.
        # Keep the readable NumPy path until an STS-specific fused kernel is
        # validated against it.
        self._fused = None

    def reset(self, q_target: np.ndarray | None = None) -> None:
        super().reset(q_target)
        # Re-seed from the new joint position on the next control substep.
        self.q_target_smooth = None

    def compute_motor_torque(self, q, dq, q_target, vin):
        """STS firmware rate limiter, then the base DC-motor equation."""
        q = np.asarray(q, dtype=np.float64)
        q_target = np.asarray(q_target, dtype=np.float64)
        if self.q_target_smooth is None:
            self.q_target_smooth = q.copy()
        max_step = self.max_velocity * self.dt
        self.q_target_smooth = np.clip(
            q_target,
            self.q_target_smooth - max_step,
            self.q_target_smooth + max_step,
        )
        return super().compute_motor_torque(q, dq, self.q_target_smooth, vin)

    def friction_budget(self, motor_torque, external_torque, dq):
        """m1 Coulomb budget; viscous damping is applied by the base pipeline."""
        return np.full_like(
            np.asarray(motor_torque, dtype=np.float64),
            self._fbase * self.friction_scale,
        )


__all__ = ["BamSTS3215Actuator"]
