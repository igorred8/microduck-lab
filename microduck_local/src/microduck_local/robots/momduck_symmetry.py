"""Left-right mirror loss for Momduck's +X-forward body convention.

Momduck now uses the same XZ mirror plane as Microduck: body forward is +X,
so reflected world/body vectors follow S = diag(1,-1,1). The joint permutation
and signs remain shared because both bodies use sign-flipped left/right home
frames; the physical mirror check in ``tests/test_momduck_symmetry.py`` pins
the observation map against MuJoCo.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..symmetry import (
    JOINT_PERM,
    JOINT_SIGN,
    OBS_PERM,
    OBS_SIGN,
    SymmetryPPO,
    _SignedPermutation,
)
from .momduck import HIP_YAW_ACTION_IDS, HIP_YAW_TARGET_LIMIT_RAD

# 0.5 -> 1.0 (2026-11-21): Jonathan asked for a stronger asymmetric-pose
# penalty while watching the walk; MOMDUCK_SYMMETRY_COEF * mirror MSE is the
# training loss term that charges exactly that.
MOMDUCK_SYMMETRY_COEF = 1.0
# v19 bounds the deployed targets themselves; the old soft actor hinge is off.
MOMDUCK_HIP_YAW_ACTION_COEF = 0.0
MOMDUCK_HIP_YAW_ACTION_RAMP = 0.5  # fraction of this run: zero -> full coef
MOMDUCK_HIP_YAW_ACTION_LIMIT_DEG = 25.0

MOMDUCK_OBS_SIGN = OBS_SIGN.copy()

assert MOMDUCK_OBS_SIGN.shape == OBS_SIGN.shape

momduck_mirror_obs = _SignedPermutation(
    OBS_PERM, MOMDUCK_OBS_SIGN, "Momduck observation")
momduck_mirror_action = _SignedPermutation(
    JOINT_PERM, JOINT_SIGN, "Momduck action")


class MomduckSymmetryPPO(SymmetryPPO):
    """Momduck mirror loss plus a gentle hip-yaw actor regularizer."""

    def __init__(self, *args,
                 hip_yaw_action_coef: float = MOMDUCK_HIP_YAW_ACTION_COEF,
                 hip_yaw_action_ramp: float = MOMDUCK_HIP_YAW_ACTION_RAMP,
                 hip_yaw_action_limit_deg: float = MOMDUCK_HIP_YAW_ACTION_LIMIT_DEG,
                 **kwargs):
        self.hip_yaw_action_coef = float(hip_yaw_action_coef)
        self.hip_yaw_action_ramp = float(hip_yaw_action_ramp)
        self.hip_yaw_action_limit_deg = float(hip_yaw_action_limit_deg)
        self._hip_yaw_run_start_steps = None
        self._hip_yaw_run_total_steps = None
        super().__init__(*args, **kwargs)

    def learn(self, total_timesteps, *args, **kwargs):
        """Start a fresh local schedule even when SB3 retains donor steps."""
        self._hip_yaw_run_start_steps = int(self.num_timesteps)
        self._hip_yaw_run_total_steps = int(total_timesteps)
        return super().learn(total_timesteps, *args, **kwargs)

    def _mirror_observation(self, obs):
        return momduck_mirror_obs(obs)

    def _mirror_action(self, action):
        return momduck_mirror_action(action)

    def _uses_action_mean(self) -> bool:
        return self.hip_yaw_action_coef > 0.0

    def _hip_yaw_action_scale(self) -> float:
        """Ramp within THIS run, not against the donor's absolute step count."""
        if self.hip_yaw_action_ramp <= 0.0:
            return 1.0
        start = self._hip_yaw_run_start_steps
        total = self._hip_yaw_run_total_steps
        if start is None or total is None or total <= 0:
            return 0.0
        elapsed_steps = max(int(self.num_timesteps) - int(start), 0)
        ramp_steps = float(total) * self.hip_yaw_action_ramp
        return float(np.clip(elapsed_steps / ramp_steps, 0.0, 1.0))

    def _hip_yaw_actor_excess(self, mean_orig: torch.Tensor) -> torch.Tensor:
        """Deterministic target overshoot, in radians, for actions 0 and 9."""
        hip_yaw = mean_orig[..., [0, 9]]
        limit = float(np.radians(self.hip_yaw_action_limit_deg))
        return F.relu(torch.abs(hip_yaw) - limit)

    def _policy_aux_losses(self, mean_orig: torch.Tensor) -> dict[str, torch.Tensor]:
        scale = self._hip_yaw_action_scale()
        if self.hip_yaw_action_coef == 0.0 or scale == 0.0:
            return {}
        excess = self._hip_yaw_actor_excess(mean_orig)
        raw = F.mse_loss(excess, torch.zeros_like(excess))
        return {
            "hip_yaw_target_limit": self.hip_yaw_action_coef * scale * raw
        }

    def _policy_aux_metrics(self, mean_orig: torch.Tensor) -> dict[str, torch.Tensor]:
        """Measure the exported (bounded) mean and raw overshoot separately."""
        with torch.no_grad():
            ids = list(HIP_YAW_ACTION_IDS)
            raw_hip_yaw = mean_orig.detach()[..., ids].reshape(-1, 2)
            hip_yaw = raw_hip_yaw.clamp(
                -HIP_YAW_TARGET_LIMIT_RAD, HIP_YAW_TARGET_LIMIT_RAD)
            mean_deg = torch.rad2deg(hip_yaw.mean(dim=0))
            raw_excess = F.relu(
                torch.abs(raw_hip_yaw) - HIP_YAW_TARGET_LIMIT_RAD)
            excess_deg = torch.rad2deg(raw_excess.mean(dim=0))
            std_deg = torch.rad2deg(self.policy.log_std.detach()[ids].exp())
            scale = hip_yaw.new_tensor(self._hip_yaw_action_scale())
        return {
            "actor_hip_yaw_mean_left_deg": mean_deg[0],
            "actor_hip_yaw_mean_right_deg": mean_deg[1],
            "actor_hip_yaw_excess_left_deg": excess_deg[0],
            "actor_hip_yaw_excess_right_deg": excess_deg[1],
            "actor_hip_yaw_std_left_deg": std_deg[0],
            "actor_hip_yaw_std_right_deg": std_deg[1],
            "actor_hip_yaw_ramp_scale": scale,
        }


__all__ = [
    "MOMDUCK_SYMMETRY_COEF",
    "MOMDUCK_HIP_YAW_ACTION_COEF",
    "MOMDUCK_HIP_YAW_ACTION_RAMP",
    "MOMDUCK_HIP_YAW_ACTION_LIMIT_DEG",
    "MOMDUCK_OBS_SIGN",
    "MomduckSymmetryPPO",
    "momduck_mirror_action",
    "momduck_mirror_obs",
]
