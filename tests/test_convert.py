"""Tests for the .pth to safetensors converter.

Verifies, per the migration plan: converted tensors are exactly equal to
the .pth state_dict, metadata records the source, and the pipeline
output is identical whether the model is loaded from the .pth or the
converted safetensors file.
"""

import pathlib

import cv2
import numpy as np
import pytest
import torch
from safetensors import safe_open

from anime_face_detector import LandmarkDetector, get_legacy_checkpoint_path
from anime_face_detector.convert import convert, extract_state_dict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_NAMES = ["faster-rcnn", "yolov3", "hrnetv2"]


@pytest.fixture(scope="session")
def converted_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("safetensors")
    for model_name in MODEL_NAMES:
        convert(
            get_legacy_checkpoint_path(model_name),
            out_dir / f"{model_name}.safetensors",
            model_name=model_name,
        )
    return out_dir


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_tensors_identical(converted_dir, model_name):
    pth_path = get_legacy_checkpoint_path(model_name)
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint)

    with safe_open(converted_dir / f"{model_name}.safetensors", framework="pt") as f:
        assert set(f.keys()) == set(state_dict.keys())
        for key in f.keys():  # noqa: SIM118 -- safe_open handles have keys() but no __iter__
            assert torch.equal(f.get_tensor(key), state_dict[key])
        metadata = f.metadata()
    assert metadata["model"] == model_name
    assert metadata["source_file"] == pth_path.name
    assert len(metadata["source_sha256"]) == 64


@pytest.mark.parametrize("detector_name", ["yolov3"])
def test_pipeline_identical(converted_dir, detector_name):
    """Outputs must be bitwise identical between .pth and safetensors loads."""
    image = cv2.imread(str(REPO_ROOT / "tests" / "golden" / "images" / "crop.png"))

    results = []
    for use_safetensors in [False, True]:
        if use_safetensors:
            landmark_path = converted_dir / "hrnetv2.safetensors"
            face_path = converted_dir / f"{detector_name}.safetensors"
        else:
            landmark_path = get_legacy_checkpoint_path("hrnetv2")
            face_path = get_legacy_checkpoint_path(detector_name)
        detector = LandmarkDetector(
            landmark_path,
            face_detector_name=detector_name,
            face_detector_checkpoint_path=face_path,
            device="cpu",
        )
        results.append(detector(image))

    assert len(results[0]) == len(results[1])
    for pred_pth, pred_st in zip(results[0], results[1], strict=True):
        np.testing.assert_array_equal(pred_pth["bbox"], pred_st["bbox"])
        np.testing.assert_array_equal(pred_pth["keypoints"], pred_st["keypoints"])
