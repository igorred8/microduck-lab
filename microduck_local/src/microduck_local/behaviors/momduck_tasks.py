"""Teach-panel recipes for Momduck.

The rewards live in ``robots/momduck_env``. The terms below are the
environment-owned scorecards shown by the viewer, not second implementations.
"""

from dataclasses import replace

from .core import *  # noqa: F401,F403 — the package's namespace cascade
from .core import Behavior, CurriculumStage, RewardTerm


def _momduck_env_owned(env):  # noqa: ARG001 — display-only task terms
    """Placeholder: Momduck rewards live in its environment, not this library."""
    return 0.0


MOMDUCK_STAND = Behavior(
    id="momduck_stand",
    suggest="stand still",
    emoji="🧍",
    title="Stand still (Momduck)",
    description=(
        "Hold the normal Momduck stance: upright, at full height, both feet "
        "planted, without drifting or crouching."
    ),
    how_it_learns=(
        "Momduck starts in STAND and practises holding DEFAULT_POSE for 20-second "
        "episodes. The score rewards nominal standing height, an upright body, "
        "both feet on the floor and a calm pose; a low crouch or a tilt past 45 "
        "degrees ends the episode, and leaning on the head or jaw is charged. "
        "This first task uses the simplified XML position actuators, with no "
        "observation noise, domain randomisation, action delay or pushes."
    ),
    keywords=(
        "stand still", "stand", "idle", "hold still", "stay", "stay still",
        "don't move", "do nothing", "at ease", "wait", "stand up straight",
    ),
    robot="momduck",
    trainer=(
        "-m", "microduck_local.train",
        "--robot", "momduck",
        "--task", "stand",
        "--actuator", "xml",
        "--lr", "0.0003",
    ),
    terms=(
        RewardTerm("still", "stays still instead of drifting", 3.0, _momduck_env_owned),
        RewardTerm("no_spin", "does not rotate or wobble the body", 1.0, _momduck_env_owned),
        RewardTerm("upright", "keeps the body upright", 2.0, _momduck_env_owned),
        RewardTerm("height", "stands at nominal height instead of crouching", 3.0,
                   _momduck_env_owned),
        RewardTerm("feet_planted", "keeps both feet on the floor", 1.0, _momduck_env_owned),
        RewardTerm("pose", "holds all joints near DEFAULT_POSE", 2.0, _momduck_env_owned),
        RewardTerm("stay_home", "stays at the episode's starting position", 1.0,
                   _momduck_env_owned),
        RewardTerm("non_foot_contact_penalty", "charged for leaning on the head, jaw or body",
                   4.0, _momduck_env_owned, is_penalty=True),
        RewardTerm("action_rate_penalty", "charged for twitchy joint commands", 0.1,
                   _momduck_env_owned, is_penalty=True),
        RewardTerm("joint_vel_penalty", "charged for fast joint motion", 0.002,
                   _momduck_env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            "hold the stance",
            1_000_000,
            {"MICRODUCK_ACTUATOR": "xml"},
            detail=(
                "Learn the nominal stance first: clean starts, zero commands and "
                "simplified XML position actuators. Noise, pushes and STS3215 BAM "
                "belong to later stages, after this stand is verified."
            ),
        ),
    ),
    default_steps=1_000_000,
    episode_s=20.0,
    success_metric=(
        "stands for 20 seconds near nominal height on multiple seeds, with both "
        "feet planted, little drift and no crouching"
    ),
)

_register(MOMDUCK_STAND)


_LOCOMOTION_TERMS = (
    RewardTerm("track_lin_vel", "tracks the requested forward speed", 4.0,
               _momduck_env_owned),
    RewardTerm("track_ang_vel", "does not rotate the body", 1.0, _momduck_env_owned),
    RewardTerm("upright", "keeps the body upright", 2.0, _momduck_env_owned),
    RewardTerm("pose", "keeps a calm pose without blocking leg swing", 0.2, _momduck_env_owned),
    RewardTerm("head_pose", "keeps the head near its commanded pose", 2.0,
               _momduck_env_owned),
    RewardTerm("feet_air_time", "pays stepping only in proportion to real forward speed",
               3.0, _momduck_env_owned),
    RewardTerm("height", "does not crouch", 3.0, _momduck_env_owned),
    RewardTerm("forward_progress", "moves forward along Momduck's +X axis", 2.0,
               _momduck_env_owned),
    RewardTerm("lateral_velocity_penalty", "charged for sideways motion", 20.0,
               _momduck_env_owned, is_penalty=True),
    RewardTerm("heading_penalty", "charged quadratically for leaving the initial course", 12.0,
               _momduck_env_owned, is_penalty=True),
    RewardTerm("yaw_rate_penalty", "charged for continuously turning left or right", 12.0,
               _momduck_env_owned, is_penalty=True),
    RewardTerm("action_rate_penalty", "charged for twitchy joint commands", 0.1,
               _momduck_env_owned, is_penalty=True),
    RewardTerm("ang_vel_xy_penalty", "charged for body rocking", 0.05,
               _momduck_env_owned, is_penalty=True),
)


MOMDUCK_FORWARD = Behavior(
    id="momduck_forward",
    suggest="walk forward",
    emoji="🚶",
    title="Walk forward (Momduck)",
    description="Walk straight along Momduck's +X axis at 0.15 m/s without crouching.",
    how_it_learns=(
        "Starts from the nominal stance and tracks a fixed 0.15 m/s forward command. "
        "Speed is measured along the MJCF's +X axis; sideways motion, course changes "
        "and a body height below 0.14 m are rejected. Loaded legs and soles are drawn "
        "back toward the stable STAND posture, while the swing leg remains free. The "
        "environment reward stays unchanged: a separate actor-only hinge penalizes "
        "the deterministic hip-yaw mean only outside ±25°, without pulling safe "
        "targets toward neutral. The first gait uses clean XML actuators without "
        "noise, pushes or randomisation."
    ),
    keywords=("walk forward", "forward", "walk straight", "straight ahead"),
    robot="momduck",
    trainer=(
        "-m", "microduck_local.train",
        "--robot", "momduck", "--task", "forward", "--actuator", "xml",
        "--lr", "0.0001",
    ),
    terms=_LOCOMOTION_TERMS + (
        RewardTerm(
            "foot_clearance",
            "pays up to 3 cm of swing-foot lift only while moving forward in single support",
            4.0, _momduck_env_owned,
        ),
        RewardTerm(
            "swing_stride",
            "pays forward swing-foot travel up to an 8 cm trailing-to-leading span",
            3.0, _momduck_env_owned,
        ),
        RewardTerm(
            "stance_pose_penalty",
            "keeps the support leg near STAND without blocking the stride",
            2.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "stance_foot_penalty",
            "keeps each loaded sole near its measured flat STAND orientation",
            1.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "stance_foot_yaw_penalty",
            "keeps loaded toes pointing along their STAND direction, not inward",
            2.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "hip_yaw_penalty",
            "holds BOTH hip_yaw joints near neutral at the stable v11 weight",
            12.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "hip_yaw_limit_penalty",
            "keeps the stable v11 soft-limit barrier beyond a 9-degree allowance",
            24.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "hip_roll_penalty",
            "keeps BOTH hips near their narrow STAND roll, including the swing leg",
            20.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "hip_roll_limit_penalty",
            "prices hip_roll excursion beyond the lateral-balance allowance",
            40.0, _momduck_env_owned, is_penalty=True,
        ),
        RewardTerm(
            "head_yaw_roll_penalty",
            "keeps the head from turning left or rolling outside the pose gradient",
            4.0, _momduck_env_owned, is_penalty=True,
        ),
    ),
    curriculum=(CurriculumStage(
        "straight forward gait", 2_000_000, {"MICRODUCK_ACTUATOR": "xml"},
        detail="Fixed 0.15 m/s along +X; no random yaw, noise or pushes.",
    ),),
    default_steps=2_000_000,
    episode_s=20.0,
    success_metric="walks at 0.15 m/s for 20 seconds without lateral drift or crouching",
)

_register(MOMDUCK_FORWARD)


_ORIGINAL_WALK_TERMS = (
    RewardTerm("track_lin_vel", "tracks the commanded body-frame speed", 4.0,
               _momduck_env_owned),
    RewardTerm("track_ang_vel", "tracks the commanded yaw rate", 1.0,
               _momduck_env_owned),
    RewardTerm("upright", "keeps the body upright", 2.0, _momduck_env_owned),
    RewardTerm("pose", "keeps a calm default pose", 1.0, _momduck_env_owned),
    RewardTerm("head_pose", "keeps the head near its commanded pose", 2.0,
               _momduck_env_owned),
    RewardTerm("feet_air_time", "pays stepping only while a command demands motion",
               5.0, _momduck_env_owned),
    RewardTerm("action_rate_penalty", "charged for twitchy joint commands", 0.1,
               _momduck_env_owned, is_penalty=True),
    RewardTerm("ang_vel_xy_penalty", "charged for body rocking", 0.05,
               _momduck_env_owned, is_penalty=True),
    RewardTerm("head_tilt_penalty", "charged for a tilted head (roll or pitch)",
               0.02, _momduck_env_owned, is_penalty=True),
    RewardTerm("forward_progress", "pays signed progress along the commanded "
               "direction; retreat pays negative and earns no air time", 2.0,
               _momduck_env_owned),
    RewardTerm("foot_separation_penalty", "charged when the feet close into a "
               "narrow stance", 2000.0, _momduck_env_owned, is_penalty=True),
)


MOMDUCK_WALK = Behavior(
    id="momduck_walk",
    suggest="walk (original recipe)",
    emoji="🚶",
    title="Walk (original duck recipe, Momduck)",
    description=(
        "The original MicroduckWalkEnv velocity-command walk, ported to Momduck's "
        "body: omnidirectional commands resampled every 5 s (2% standing, 20% "
        "forward, turns and omnidirectional otherwise), observation noise, action "
        "delay, domain randomisation and velocity pushes ON — the conditions the "
        "duck actually learned to walk in by ~600k steps."
    ),
    how_it_learns=(
        "Unlike momduck_forward's fixed 0.15 m/s sterile stage, here the world "
        "itself forces locomotion: most commands demand motion, pushes knock the "
        "body off balance so stepping gets sampled and rewarded, and domain "
        "randomisation prevents overfitting to one clean plant. XML actuators are "
        "kept constant from the previous arms, so this arm changes the RECIPE and "
        "only the recipe; BAM belongs to a later verification stage."
    ),
    keywords=(
        "walk", "original walk", "original recipe", "classic walk",
        "velocity walk", "walk original", "omnidirectional walk",
    ),
    robot="momduck",
    trainer=(
        "-m", "microduck_local.train",
        "--robot", "momduck", "--task", "walk", "--actuator", "xml",
    ),
    terms=_ORIGINAL_WALK_TERMS,
    curriculum=(CurriculumStage(
        "original velocity-command walk", 3_000_000, {"MICRODUCK_ACTUATOR": "xml"},
        detail=(
            "The duck's own recipe: command mixture, pushes, domain randomisation, "
            "observation noise and action delay all ON. 3M steps, like train-walk."
        ),
    ),),
    default_steps=3_000_000,
    episode_s=20.0,
    success_metric=(
        "tracks omnidirectional velocity commands while staying upright; "
        "first walking expected by ~600k steps, as in the duck's original recipe"
    ),
)

_register(MOMDUCK_WALK)


MOMDUCK_WALK_BAM = replace(
    MOMDUCK_WALK,
    id="momduck_walk_bam",
    suggest="walk with BAM STS3215",
    title="Walk with BAM STS3215 (Momduck)",
    keywords=("walk with bam", "bam walk", "sts3215 walk", "bam"),
    description=(
        "Same original velocity-command walk as momduck_walk, but with the real "
        "STS3215 BAM actuator instead of the simplified XML position actuators. "
        "For fine-tuning alpha_walking.onnx on Momduck's actual servo physics."
    ),
    how_it_learns=(
        "Identical command mixture, pushes, domain randomisation, observation "
        "noise and action delay to momduck_walk; only the actuator changes: "
        "BAM STS3215 m1 with kp_fw=32, error_gain=0.166, vin 7.0-8.2 V and "
        "the firmware rate limiter."
    ),
    trainer=(
        "-m", "microduck_local.train",
        "--robot", "momduck", "--task", "walk", "--actuator", "bam", "--lr", "5e-5",
        "--device", "cuda",
    ),
    curriculum=(CurriculumStage(
        "original velocity-command walk with BAM STS3215", 3_000_000,
        {"MICRODUCK_ACTUATOR": "bam"},
        detail="Same recipe as momduck_walk, but with the real servo model.",
    ),),
)
_register(MOMDUCK_WALK_BAM)


MOMDUCK_RUN = Behavior(
    id="momduck_run",
    suggest="run forward",
    emoji="🏃",
    title="Run: adaptive speed (Momduck)",
    description=(
        "Start at 0.4 m/s and add 0.2 m/s only after stable full episodes, up to 2.0 m/s."
    ),
    how_it_learns=(
        "All environments start at 0.4 m/s along Momduck's +X axis. The fleet advances "
        "by 0.2 m/s only after 12 consecutive full 20-second episodes with mean speed "
        "within 10% of target, speed variation within 15%, little lateral motion and no "
        "falls. A failure breaks the confirmation streak but never lowers the target. "
        "run_status.json records the current target and the maximum confirmed speed."
    ),
    keywords=("run forward", "run", "running", "maximum speed", "speed curriculum"),
    robot="momduck",
    trainer=(
        "-m", "microduck_local.train",
        "--robot", "momduck", "--task", "run", "--actuator", "xml",
        "--lr", "0.0001",
    ),
    terms=_LOCOMOTION_TERMS,
    curriculum=(CurriculumStage(
        "0.4 + 0.2 m/s after stable episodes",
        30_000_000,
        {"MICRODUCK_ACTUATOR": "xml"},
        detail=(
            "Performance-gated, not time-gated: 12 stable full episodes per level; "
            "2.0 m/s cap; every episode and promotion is logged."
        ),
    ),),
    default_steps=30_000_000,
    episode_s=20.0,
    success_metric=(
        "holds each target for 12 consecutive full episodes before advancing; "
        "the morning result is max_confirmed_speed, not merely the latest command"
    ),
)

_register(MOMDUCK_RUN)
