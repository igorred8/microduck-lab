"""Import alpha_walking.onnx actor into an SB3 PPO donor for warm-start.

Creates runs/momduck-alpha-warmstart/{model.zip, vecnormalize.pkl, run.json, record.json}.
Architectures match exactly: 61 -> 512 -> 256 -> 128 -> 14, ELU.
Critic is fresh random; obs normalizer comes from the ONNX buffers.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from microduck_local.robots.momduck_env import MomduckWalkEnv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "momduck-alpha-warmstart"
OUT.mkdir(parents=True, exist_ok=True)

m = onnx.load(str(ROOT.parent / "microduck" / "policies" / "alpha_walking.onnx"))
W = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
print("ONNX tensors:", {k: v.shape for k, v in W.items() if len(v.shape) >= 1})

venv = DummyVecEnv([lambda: MomduckWalkEnv(
    actuator_force="bam", obs_noise=False, domain_rand=False,
    action_delay=False, push_robot=False)])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=np.inf)
model = PPO(
    "MlpPolicy", venv,
    policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                       activation_fn=torch.nn.ELU, log_std_init=-1.0),
    learning_rate=1e-4,
    seed=0,
)

with torch.no_grad():
    actor = model.policy
    for onnx_name, torch_mod in (
        ("mlp.0", actor.mlp_extractor.policy_net[0]),
        ("mlp.2", actor.mlp_extractor.policy_net[2]),
        ("mlp.4", actor.mlp_extractor.policy_net[4]),
        ("mlp.6", actor.action_net),
    ):
        torch_mod.weight.copy_(torch.from_numpy(W[onnx_name + ".weight"]))
        torch_mod.bias.copy_(torch.from_numpy(W[onnx_name + ".bias"]))
    # Orthogonal init of log_std is not in the ONNX; leave log_std_init=-1.0.
    actor.log_std.fill_(-1.0)

obs_mean = W["obs_normalizer._mean"].reshape(-1)
obs_std = W["onnx::Div_24"].reshape(-1)
venv.obs_rms.mean = obs_mean
venv.obs_rms.var = obs_std ** 2
venv.obs_rms.count = 1e6  # freeze-ish: do not re-estimate on the first rollouts

model.save(str(OUT / "model.zip"))
venv.save(str(OUT / "vecnormalize.pkl"))

(OUT / "run.json").write_text(json.dumps({
    "robot": "momduck", "task": "walk", "contract_id": "momduck-61-v2",
    "init_from": "onnx:alpha_walking",
    "actor_source": "alpha_walking.onnx (mlp.0/2/4/6 + obs normalizer)",
    "critic": "fresh random", "log_std_init": -1.0,
    "learning_rate": 1e-4,
    "actuator_force": "bam",
}, indent=2))
(OUT / "record.json").write_text(json.dumps({
    "title": "alpha_walking imported (warm-start donor)",
    "description": "alpha_walking.onnx actor weights in SB3 PPO, critic fresh. "
                   "For fine-tuning on momduck with BAM STS3215.",
    "run_name": "momduck-alpha-warmstart",
    "robot": "momduck", "contract_id": "momduck-61-v2", "task": "walk",
}, indent=2))
print("saved donor to", OUT)
