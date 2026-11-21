"""Momduck: a Microduck-compatible walker with its own assets and policy contract."""

from __future__ import annotations

import os
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .. import contract as C
from .microduck import MicroduckBody, _obs_slots
from .policy_contract import PolicyContract, declare
from .spec import Effector

# src/microduck_local/robots/momduck.py -> microduck_local project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "momduck"
MODEL_PATH = MODEL_DIR / "model.xml"
SCENE_PATH = MODEL_DIR / "scene_walk.xml"
MESH_DIR = PROJECT_ROOT / "meshes" / "momduck"
CONTRACT_ID = "momduck-61-v2"
HIP_YAW_ACTION_IDS = (0, 9)
HIP_YAW_TARGET_LIMIT_DEG = 10.0
HIP_YAW_TARGET_LIMIT_RAD = float(np.radians(HIP_YAW_TARGET_LIMIT_DEG))
HIP_YAW_EXPLORATION_STD_DEG = 10.0


def bound_hip_yaw_actions(action: np.ndarray) -> np.ndarray:
    """Return a copy with both DEFAULT_POSE-relative yaw targets bounded."""
    out = np.array(action, dtype=np.float32, copy=True)
    out[..., HIP_YAW_ACTION_IDS] = np.clip(
        out[..., HIP_YAW_ACTION_IDS],
        -HIP_YAW_TARGET_LIMIT_RAD,
        HIP_YAW_TARGET_LIMIT_RAD,
    )
    return out


def _scene_fn() -> Path:
    """The walking scene, including Momduck, its floor and its light."""
    return SCENE_PATH


@lru_cache(maxsize=1)
def _visual_scene() -> dict:
    """Momduck's own visual meshes, at a resolution suitable for a small CAD model."""
    import mujoco

    from .g1 import extract_visual_scene

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return extract_visual_scene(
        model,
        group=2,
        vert_scale=0.0001,
        normals=True,
    )


class MomduckBody(MicroduckBody):
    """Reuse the compatible walker, never Microduck's assets or shipped policies."""

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Bound yaw before `last_action`, exploration target, and deployment."""
        return bound_hip_yaw_actions(action)

    def action_metadata(self) -> dict:
        return {
            "hip_yaw_action_indices": list(HIP_YAW_ACTION_IDS),
            "hip_yaw_target_limit_deg": HIP_YAW_TARGET_LIMIT_DEG,
            "hip_yaw_initial_exploration_std_deg": HIP_YAW_EXPLORATION_STD_DEG,
            "last_action": "bounded policy output",
        }

    def contract(self) -> PolicyContract:
        """The shared 61-float layout, versioned separately for Momduck."""
        return declare(
            self,
            id=CONTRACT_ID,
            rate_hz=round(1.0 / C.CTRL_DT, 6),
            slots=_obs_slots(self.num_joints),
            deploy=(
                "lab contract: 61 observations and 14 position-target deltas at "
                "50 Hz; actions 0/9 and their last_action slots are bounded to "
                "+/-10 deg; hardware deployment requires a matching Momduck runtime"
            ),
        )

    def ready(self) -> bool:
        return MODEL_PATH.is_file() and SCENE_PATH.is_file() and MESH_DIR.is_dir()

    def look(self) -> str:
        # Draw Momduck's own MJCF colours through the generic CAD material.
        return "generic"

    def visual_scene(self) -> dict:
        return _visual_scene()

    def shipped_policies(self) -> tuple[dict, ...]:
        # Pollen's policies belong to Microduck, even though their widths match.
        return ()

    def env_class(self, task: str = "walk") -> type:
        if task in (None, "", "walk"):
            from .momduck_env import MomduckWalkEnv
            return MomduckWalkEnv
        if task == "forward":
            from .momduck_env import MomduckForwardEnv
            return MomduckForwardEnv
        if task == "run":
            from .momduck_env import MomduckRunEnv
            return MomduckRunEnv
        if task == "stand":
            from .momduck_env import MomduckStandEnv
            return MomduckStandEnv
        raise SystemExit(
            f"unknown task {task!r} for Momduck; available tasks: walk, forward, run, stand"
        )

    def train_env_kwargs(self, args: Any) -> dict:
        """Use CLI/environment actuator selection, defaulting to STS3215 BAM."""
        actuator = (
            getattr(args, "actuator", None)
            or os.environ.get("MICRODUCK_ACTUATOR")
            or "bam"
        )
        kw = {"actuator_force": actuator}
        task = getattr(args, "task", "walk") or "walk"
        if task in ("stand", "forward", "run"):
            # Record the nominal curriculum in run.json so viewer replay uses
            # the same clean initial conditions as training.
            kw.update(
                max_episode_s=20.0,
                obs_noise=False,
                domain_rand=False,
                action_delay=False,
                push_robot=False,
            )
        return kw

    def attach(
        self,
        spec: Any,
        prefix: str,
        frame: Any,
        collision: str = "walk",
    ) -> None:
        """Attach the robot-only MJCF; the floor and light live in scene_walk.xml."""
        import mujoco

        if collision != "walk":
            raise ValueError("Momduck currently provides only the 'walk' collision model")
        robot = mujoco.MjSpec.from_file(str(MODEL_PATH))
        spec.attach(robot, prefix=prefix, frame=frame)

    def make_sensors(
        self,
        model: Any,
        prefix: str,
        *,
        presets: Any,
        targets: Any,
        seed: Any,
    ) -> dict:
        # This MJCF has IMU sensors, but no `tof` or `head_camera` sites.
        return {}


def _microduck_fields(**overrides: Any) -> dict[str, Any]:
    """Copy compatible RobotSpec data, while constructing a MomduckBody."""
    values = {field.name: getattr(C.MICRODUCK, field.name)
              for field in fields(C.MICRODUCK)}
    unknown = overrides.keys() - values.keys()
    if unknown:
        raise TypeError(f"Unknown RobotSpec fields: {sorted(unknown)}")
    values.update(overrides)
    return values


EFFECTORS = (
    Effector("left_foot", "left foot", "ankle_left", "foot", "sole"),
    Effector("right_foot", "right foot", "ankle_r", "foot", "sole"),
    Effector("head", "head", "jaw_soft", "head", (0.0, 0.0, 0.0)),
)

BODY = MomduckBody(**_microduck_fields(
    id="momduck",
    title="Momduck",
    noun="momduck",
    scene_fn=_scene_fn,
    stand_keyframe="STAND",
    gyro_sensor="angular-velocity",
    effectors=EFFECTORS,
))

MOMDUCK_SPEC = BODY

__all__ = ["BODY", "MOMDUCK_SPEC", "MomduckBody"]
