"""Score Momduck's stand on deterministic 20-second episodes.

    uv run python scripts/probe_momduck_stand.py
    uv run python scripts/probe_momduck_stand.py runs/<run>
    uv run python scripts/probe_momduck_stand.py runs/<run> --all-checkpoints
    uv run python scripts/probe_momduck_stand.py runs/<run>/policy.onnx --episodes 24

With no target this prices zero action, the floor every learned policy must
beat. A run directory is exported afresh into a temporary directory. Episodes
are paired by seed across every target.
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
from microduck_local.robots.momduck_env import MomduckStandEnv


def make_env(seed: int, seconds: float) -> MomduckStandEnv:
    return MomduckStandEnv(
        actuator_force="xml",
        seed=seed,
        max_episode_s=seconds,
        obs_noise=False,
        domain_rand=False,
        action_delay=False,
        push_robot=False,
    )


def episode_metrics(env: MomduckStandEnv, seed: int, policy) -> dict:
    obs, _ = env.reset(seed=seed)
    gravity0 = obs[3:6].astype(float).copy()
    heights, speeds, spins, tilts, planted, rewards = [], [], [], [], [], []
    while True:
        action = (np.zeros(env.action_space.shape, dtype=np.float32)
                  if policy is None else policy(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        heights.append(float(env._trunk_xpos[2]))
        speeds.append(float(np.linalg.norm(env.body_lin_vel())))
        spins.append(float(np.linalg.norm(env._gyro)))
        gravity = env._projected_gravity()
        tilts.append(float(np.degrees(np.arctan2(
            np.linalg.norm(gravity[:2]), -gravity[2]))))
        contacts = env._foot_contacts()
        planted.append(float(contacts["left"] and contacts["right"]))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    drift = float(np.linalg.norm(env._trunk_xpos[:2] - env._spawn_xy))
    height_error = float(abs(np.mean(heights) - env.stand_z))
    completed = bool(truncated and not terminated)
    goal = bool(completed and height_error <= 0.01 and drift <= 0.05
                and np.mean(planted) >= 0.99)
    return {
        "seed": seed,
        "completed": completed,
        "goal": goal,
        "steps": int(env.step_count),
        "reward": float(np.sum(rewards)),
        "height_m": float(np.mean(heights)),
        "height_error_m": height_error,
        "final_drift_m": drift,
        "speed_mps": float(np.mean(speeds)),
        "spin_radps": float(np.mean(spins)),
        "tilt_deg": float(np.mean(tilts)),
        "planted_fraction": float(np.mean(planted)),
        "initial_gravity": gravity0.tolist(),
    }


def score(name: str, policy, episodes: int, seed: int, seconds: float) -> dict:
    rows = []
    for ep in range(episodes):
        env = make_env(seed + ep, seconds)
        try:
            rows.append(episode_metrics(env, seed + ep, policy))
        finally:
            env.close()
    mean_keys = ("height_m", "height_error_m", "final_drift_m", "speed_mps",
                 "spin_radps", "tilt_deg", "planted_fraction", "reward")
    result = {
        "name": name,
        "episodes": rows,
        "completed": sum(r["completed"] for r in rows),
        "goal": sum(r["goal"] for r in rows),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "initial_gravity": rows[0]["initial_gravity"],
        **{k: float(np.mean([r[k] for r in rows])) for k in mean_keys},
    }
    print(
        f"{name}: full {result['completed']}/{episodes}, "
        f"goal {result['goal']}/{episodes}, steps {result['mean_steps']:.0f}, "
        f"height {result['height_m']:.4f} m "
        f"(err {result['height_error_m']:.4f}), "
        f"drift {result['final_drift_m']:.4f} m, "
        f"speed {result['speed_mps']:.4f} m/s, "
        f"spin {result['spin_radps']:.4f} rad/s, "
        f"tilt {result['tilt_deg']:.2f} deg, "
        f"planted {100 * result['planted_fraction']:.1f}%, "
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
    with tempfile.TemporaryDirectory(prefix="momduck-stand-probe-") as tmp:
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
