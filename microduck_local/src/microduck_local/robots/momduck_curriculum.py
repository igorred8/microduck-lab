"""Synchronised, performance-gated speed curriculum for MomduckRunEnv."""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from .momduck_env import MomduckRunEnv


class MomduckRunCurriculum(BaseCallback):
    """Raise the fleet target only after twelve consecutive stable episodes."""

    PROMOTE_EPISODES = 12
    MEAN_SPEED_TOL = 0.10       # episode mean within +/-10% of target
    SPEED_STD_REL = 0.15        # stride-to-stride variation <= 15% of target
    MAX_MEAN_LATERAL = 0.05     # m/s, averaged over the full episode

    def __init__(self, run_dir: Path):
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.target_speed = MomduckRunEnv.START_SPEED
        self.confirmed_speed = 0.0
        self.max_confirmed_speed = 0.0
        self.promotions = 0
        self.episodes = 0
        self._window = deque(maxlen=self.PROMOTE_EPISODES)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _event(self, kind: str, **payload) -> None:
        record = {
            "event": kind,
            "time": self._now(),
            "steps": int(self.num_timesteps),
            **payload,
        }
        with (self.run_dir / "run_curriculum.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_status(self) -> None:
        payload = {
            "updated": self._now(),
            "steps": int(self.num_timesteps),
            "target_speed": self.target_speed,
            "confirmed_speed": self.confirmed_speed,
            "max_confirmed_speed": self.max_confirmed_speed,
            "promotions": self.promotions,
            "episodes": self.episodes,
            "consecutive_successes": sum(self._window),
            "required_successes": self.PROMOTE_EPISODES,
            "max_speed_cap": MomduckRunEnv.MAX_SPEED,
        }
        tmp = self.run_dir / "run_status.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.run_dir / "run_status.json")

    def _on_training_start(self) -> None:
        self._event("start", target_speed=self.target_speed,
                    required_successes=self.PROMOTE_EPISODES)
        self._write_status()

    def _stable(self, m: dict) -> bool:
        target = float(m["target_speed"])
        return bool(
            m["completed"]
            and abs(float(m["mean_speed"]) - target)
                <= self.MEAN_SPEED_TOL * target
            and float(m["speed_std"]) <= self.SPEED_STD_REL * target
            and float(m["mean_abs_lateral"]) <= self.MAX_MEAN_LATERAL
        )

    def _record_episode(self, metrics: dict) -> None:
        target = float(metrics["target_speed"])
        # Episodes already in flight during a promotion still finish at their
        # original target. They are logged, but cannot promote the new level.
        if abs(target - self.target_speed) > 1e-9:
            self._event("ignored_previous_target", **metrics)
            return

        self.episodes += 1
        stable = self._stable(metrics)
        self._window.append(stable)
        self._event("episode", stable=stable, **metrics)

        if len(self._window) < self.PROMOTE_EPISODES or not all(self._window):
            self._write_status()
            return

        previous = self.target_speed
        self.confirmed_speed = previous
        self.max_confirmed_speed = max(self.max_confirmed_speed, previous)
        self._window.clear()

        if previous >= MomduckRunEnv.MAX_SPEED:
            self._event("maximum_confirmed", speed=previous)
            self._write_status()
            return

        self.target_speed = round(min(
            previous + MomduckRunEnv.SPEED_STEP, MomduckRunEnv.MAX_SPEED), 1)
        self.promotions += 1
        self.model.get_vec_normalize_env().env_method(
            "set_run_target_speed", self.target_speed)
        self._event(
            "promotion",
            from_speed=previous,
            to_speed=self.target_speed,
            confirmed_speed=previous,
            episodes_confirmed=self.PROMOTE_EPISODES,
        )
        print(f"[momduck-run] confirmed {previous:.1f} m/s; "
              f"next target {self.target_speed:.1f} m/s", flush=True)
        self._write_status()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            metrics = info.get("momduck_run")
            if metrics:
                self._record_episode(metrics)
        return True

    def _on_training_end(self) -> None:
        self._write_status()
