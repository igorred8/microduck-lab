"""Score Momduck's straight forward walk on deterministic 20-second episodes.

    uv run python scripts/probe_momduck_forward.py
    uv run python scripts/probe_momduck_forward.py runs/<run>
    uv run python scripts/probe_momduck_forward.py runs/<run> --all-checkpoints
    uv run python scripts/probe_momduck_forward.py runs/<run>/policy.onnx \
        --episodes 24

With no target this prices zero action, the floor every learned gait must beat.
A run directory is exported afresh into a temporary directory. Episodes are
paired by seed across every target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from microduck_local.export_onnx import export
from microduck_local.robots.momduck_env import MomduckForwardEnv


def make_env(seed: int, seconds: float) -> MomduckForwardEnv:
    return MomduckForwardEnv(
        actuator_force="xml",
        seed=seed,
        max_episode_s=seconds,
        obs_noise=False,
        domain_rand=False,
        action_delay=False,
        push_robot=False,
    )


def _non_foot_floor_contact(env: MomduckForwardEnv) -> bool:
    foot_bodies = {
        int(env.model.geom_bodyid[gid]) for gid in env.foot_geoms.values()
    }
    for i in range(env.data.ncon):
        g1 = int(env.data.contact.geom1[i])
        g2 = int(env.data.contact.geom2[i])
        if env.floor_geom not in (g1, g2):
            continue
        other = g2 if g1 == env.floor_geom else g1
        if int(env.model.geom_bodyid[other]) not in foot_bodies:
            return True
    return False


def _heading_error_rad(env: MomduckForwardEnv) -> float:
    current = env._trunk_xmat.reshape(3, 3) @ env.BODY_FORWARD
    current[2] = 0.0
    norm = float(np.linalg.norm(current))
    if norm <= 1e-9:
        return float(np.pi / 2.0)
    current /= norm
    return float(np.arctan2(
        current @ env._spawn_side,
        np.clip(current @ env._spawn_forward, -1.0, 1.0)))


def episode_metrics(env: MomduckForwardEnv, seed: int, policy) -> dict:
    obs, _ = env.reset(seed=seed)
    start_xy = np.asarray(env._trunk_xpos[:2], float).copy()
    forward3 = np.asarray(env._spawn_forward, float)
    side3 = np.asarray(env._spawn_side, float)
    forward = forward3[:2]

    heights, fwd_speeds, lat_speeds, headings = [], [], [], []
    contact_states, nonfoot_frames = [], 0
    clearances, single_support, rewards = [], 0, []
    prev_state = None
    support_changes = 0

    while True:
        action = (np.zeros(env.action_space.shape, dtype=np.float32)
                  if policy is None else policy(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        world_v = np.asarray(env._qvel_base, float)
        heights.append(float(env._trunk_xpos[2]))
        fwd_speeds.append(float(world_v @ forward3))
        lat_speeds.append(float(world_v @ side3))
        headings.append(abs(_heading_error_rad(env)))
        rewards.append(float(reward))
        nonfoot_frames += int(_non_foot_floor_contact(env))

        contacts = env._foot_contacts()
        state = (bool(contacts["left"]), bool(contacts["right"]))
        contact_states.append(state)
        if prev_state is not None and state != prev_state:
            support_changes += 1
        prev_state = state
        if state[0] != state[1]:
            single_support += 1
            swing = "right" if state[0] else "left"
            lift = (float(env.data.geom_xpos[env.foot_geoms[swing]][2])
                    - env.foot_stand_z[swing])
            clearances.append(max(0.0, lift))

        if terminated or truncated:
            break

    delta_xy = np.asarray(env._trunk_xpos[:2], float) - start_xy
    completed = bool(truncated and not terminated)
    mean_fwd = float(np.mean(fwd_speeds))
    goal = bool(
        completed
        and 0.12 <= mean_fwd <= 0.18
        and float(delta_xy @ forward) >= 2.4
        and float(np.sqrt(np.mean(np.square(lat_speeds)))) <= 0.03
        and float(np.sqrt(np.mean(np.square(headings)))) <= 0.25
        and min(heights) >= env.FORWARD_FALL_HEIGHT
        and nonfoot_frames == 0
        and support_changes >= 8
        and 0.05 <= single_support / max(1, len(contact_states)) <= 0.85
    )
    return {
        "seed": seed,
        "completed": completed,
        "goal": goal,
        "steps": int(env.step_count),
        "reward": float(np.sum(rewards)),
        "mean_forward_speed_mps": mean_fwd,
        "net_forward_distance_m": float(delta_xy @ forward),
        "lateral_rms_mps": float(np.sqrt(np.mean(np.square(lat_speeds)))),
        "heading_rmse_rad": float(np.sqrt(np.mean(np.square(headings)))),
        "min_trunk_z_m": float(min(heights)),
        "mean_trunk_z_m": float(np.mean(heights)),
        "nonfoot_floor_frames": int(nonfoot_frames),
        "support_changes": int(support_changes),
        "single_support_fraction": float(
            single_support / max(1, len(contact_states))),
        "mean_swing_clearance_m": float(np.mean(clearances)) if clearances else 0.0,
        "target_speed_mps": float(env.FORWARD_SPEED),
    }


def score(name: str, policy, episodes: int, seed: int, seconds: float) -> dict:
    rows = []
    for ep in range(episodes):
        env = make_env(seed + ep, seconds)
        try:
            rows.append(episode_metrics(env, seed + ep, policy))
        finally:
            env.close()
    mean_keys = (
        "reward", "mean_forward_speed_mps", "net_forward_distance_m",
        "lateral_rms_mps", "heading_rmse_rad", "min_trunk_z_m",
        "mean_trunk_z_m", "single_support_fraction", "mean_swing_clearance_m",
    )
    result = {
        "name": name,
        "episodes": rows,
        "completed": sum(r["completed"] for r in rows),
        "goal": sum(r["goal"] for r in rows),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_nonfoot_floor_frames": float(np.mean(
            [r["nonfoot_floor_frames"] for r in rows])),
        "mean_support_changes": float(np.mean(
            [r["support_changes"] for r in rows])),
        **{k: float(np.mean([r[k] for r in rows])) for k in mean_keys},
    }
    print(
        f"{name}: full {result['completed']}/{episodes}, "
        f"goal {result['goal']}/{episodes}, steps {result['mean_steps']:.0f}, "
        f"fwd {result['mean_forward_speed_mps']:.3f} m/s "
        f"(net {result['net_forward_distance_m']:.2f} m), "
        f"lat_rms {result['lateral_rms_mps']:.3f} m/s, "
        f"head_rmse {np.degrees(result['heading_rmse_rad']):.1f} deg, "
        f"min_z {result['min_trunk_z_m']:.3f} m, "
        f"nonfoot {result['mean_nonfoot_floor_frames']:.1f} frames, "
        f"support_changes {result['mean_support_changes']:.1f}, "
        f"single {100 * result['single_support_fraction']:.1f}%, "
        f"clearance {1000 * result['mean_swing_clearance_m']:.1f} mm, "
        f"reward {result['reward']:.0f}", flush=True)
    return result


def onnx_policy(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def act(obs):
        return session.run(None, {input_name: obs[None]})[0][0].astype(np.float32)

    return act


def checkpoint_paths(run: Path):
    pairs = []
    for model in sorted(run.glob("checkpoints/model_*_steps.zip"),
                        key=lambda p: int(p.stem.split("_")[1])):
        steps = model.stem.split("_")[1]
        normalizer = run / "checkpoints" / f"model_vecnormalize_{steps}_steps.pkl"
        if not normalizer.is_file():
            raise FileNotFoundError(f"missing normalizer beside {model}")
        pairs.append((f"{run.name}-{steps}", model, normalizer))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", type=Path,
                        help="ONNX files or run directories")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_intermixed_args()
    if args.episodes < 1 or args.seconds <= 0:
        parser.error("--episodes and --seconds must be positive")

    results = []
    if not args.targets:
        results.append(score("zero-action", None, args.episodes,
                             args.seed, args.seconds))
    with tempfile.TemporaryDirectory(prefix="momduck-forward-probe-") as tmp:
        scratch = Path(tmp)
        for target in args.targets:
            if target.suffix == ".onnx":
                print(f"policy sha256 {hashlib.sha256(target.read_bytes()).hexdigest()}")
                results.append(score(target.name, onnx_policy(target),
                                     args.episodes, args.seed, args.seconds))
                continue
            run = target
            pairs = checkpoint_paths(run) if args.all_checkpoints else []
            if not pairs:
                pairs = [(f"{run.name}-final", run / "model.zip",
                          run / "vecnormalize.pkl")]
            for name, model, normalizer in pairs:
                out = scratch / f"{name}.onnx"
                export(run, out, model_path=model, vn_path=normalizer)
                print(f"policy sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
                results.append(score(name, onnx_policy(out), args.episodes,
                                     args.seed, args.seconds))
    report = {"episodes": args.episodes, "seed": args.seed,
              "seconds": args.seconds, "results": results}
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
