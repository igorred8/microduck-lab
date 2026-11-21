"""Momduck v2 action bounds: training, last_action, and deterministic export."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from microduck_local import contract as C
from microduck_local.export_onnx import OnnxWalkPolicy
from microduck_local.robots.momduck import (
    BODY,
    HIP_YAW_ACTION_IDS,
    HIP_YAW_TARGET_LIMIT_DEG,
    HIP_YAW_TARGET_LIMIT_RAD,
    bound_hip_yaw_actions,
)


def test_momduck_bounds_only_both_hip_yaw_targets_and_is_idempotent():
    action = np.linspace(-1.0, 1.0, C.NUM_JOINTS, dtype=np.float32)
    action[[0, 9]] = np.radians([-30.0, 30.0])
    original = action.copy()

    got = BODY.process_action(action)
    assert got is not action
    np.testing.assert_array_equal(action, original)  # caller's array is not mutated
    assert np.degrees(got[0]) == pytest.approx(-HIP_YAW_TARGET_LIMIT_DEG, abs=1e-5)
    assert np.degrees(got[9]) == pytest.approx(HIP_YAW_TARGET_LIMIT_DEG, abs=1e-5)
    other = [i for i in range(C.NUM_JOINTS) if i not in HIP_YAW_ACTION_IDS]
    np.testing.assert_array_equal(got[other], action[other])
    np.testing.assert_array_equal(BODY.process_action(got), got)


def test_momduck_bound_is_exact_inside_and_at_the_limits():
    action = np.zeros(C.NUM_JOINTS, dtype=np.float32)
    action[[0, 9]] = np.radians([-10.0, 5.0])
    got = bound_hip_yaw_actions(action)
    np.testing.assert_allclose(got, action, rtol=0, atol=2e-8)
    assert abs(float(got[0])) <= HIP_YAW_TARGET_LIMIT_RAD
    assert abs(float(got[9])) <= HIP_YAW_TARGET_LIMIT_RAD


def test_other_bodies_keep_the_raw_action_identity():
    action = np.linspace(-4.0, 4.0, C.NUM_JOINTS, dtype=np.float32)
    assert C.MICRODUCK.process_action(action) is action
    assert C.MICRODUCK.action_metadata() == {}


class _FixedActions(torch.nn.Module):
    def forward(self, features):
        raw = torch.zeros(C.NUM_JOINTS, dtype=features.dtype)
        raw[0], raw[9] = np.radians(-30.0), np.radians(30.0)
        other = [i for i in range(C.NUM_JOINTS) if i not in HIP_YAW_ACTION_IDS]
        raw[other] = torch.linspace(-0.4, 0.4, len(other))
        return raw.expand(features.shape[0], -1) + 0.0 * features[:, :1]


class _ActorOnly(torch.nn.Module):
    def forward_actor(self, features):
        return features


class _DummyPolicy(torch.nn.Module):
    action_space = SimpleNamespace(shape=(C.NUM_JOINTS,))

    def __init__(self):
        super().__init__()
        self.features_extractor = torch.nn.Identity()
        self.mlp_extractor = _ActorOnly()
        self.action_net = _FixedActions()

    def extract_features(self, obs, features_extractor):
        return features_extractor(obs)


def _wrapper(contract_id=None):
    return OnnxWalkPolicy(
        _DummyPolicy(), np.zeros(C.OBS_DIM), np.ones(C.OBS_DIM), 100.0,
        contract_id=contract_id,
    ).eval()


def test_onnx_wrapper_matches_the_training_transform_and_default_is_noop():
    obs = torch.zeros(2, C.OBS_DIM)
    with torch.no_grad():
        raw = _wrapper()(obs)
        bounded = _wrapper("momduck-61-v2")(obs)
    np.testing.assert_allclose(
        bounded.numpy(), BODY.process_action(raw[0].numpy())[None].repeat(2, axis=0),
        rtol=0, atol=1e-7,
    )
    assert float(raw[:, 0].abs().max()) > HIP_YAW_TARGET_LIMIT_RAD
    assert float(raw[:, 9].abs().max()) > HIP_YAW_TARGET_LIMIT_RAD


def test_exported_momduck_graph_contains_the_bound(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path / "momduck-v2.onnx"
    wrapper = _wrapper("momduck-61-v2")
    dummy = torch.zeros(1, C.OBS_DIM)
    torch.onnx.export(
        wrapper, (dummy,), str(path), input_names=["obs"],
        output_names=["actions"], opset_version=17, dynamo=False,
    )
    got = ort.InferenceSession(str(path)).run(
        ["actions"], {"obs": np.zeros((1, C.OBS_DIM), np.float32)})[0][0]
    np.testing.assert_allclose(got, BODY.process_action(got), rtol=0, atol=1e-7)
    assert np.degrees(got[0]) == pytest.approx(-10.0, abs=1e-4)
    assert np.degrees(got[9]) == pytest.approx(10.0, abs=1e-4)
