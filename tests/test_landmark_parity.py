"""Landmark parity against the golden outputs of the mm*-based implementation.

Tolerances: coordinates must match within 0.5 px. Scores and raw heatmap
values are compared at 1e-2 instead of the originally planned 1e-3: the
goldens were produced with torch 1.13, whose CPU convolutions differ
numerically from those of torch 2.x. This shifts heatmap values by up
to ~5e-3 while leaving decoded coordinates essentially unchanged
(<0.3 px on the golden images). Layer statistics agree to ~1e-5 (see
tests/golden/intermediates_hrnetv2.npz).
"""

import pathlib

import cv2
import numpy as np
import pytest

from anime_face_detector import get_legacy_checkpoint_path
from anime_face_detector._landmark import load_landmark_model

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

XY_TOLERANCE = 0.5
SCORE_TOLERANCE = 1e-2
HEATMAP_TOLERANCE = 1e-2

IMAGE_PATHS = {
    "input": REPO_ROOT / "assets" / "input.jpg",
    "crop": GOLDEN_DIR / "images" / "crop.png",
}


@pytest.fixture(scope="session")
def model():
    # load_landmark_model() uses strict=True, so this fixture also acts as
    # the state_dict key compatibility check.
    return load_landmark_model(str(get_legacy_checkpoint_path("hrnetv2")), device="cpu")


@pytest.fixture(scope="session")
def golden():
    with np.load(GOLDEN_DIR / "landmarks.npz") as data:
        return dict(data)


@pytest.mark.parametrize("image_name", ["crop", "input"])
@pytest.mark.parametrize("flip_test", [True, False])
def test_landmark_parity(model, golden, image_name, flip_test):
    image = cv2.imread(str(IMAGE_PATHS[image_name]))
    boxes = golden[f"{image_name}/boxes"]
    keypoints = model.detect(image, boxes, flip_test=flip_test)

    key = "flip" if flip_test else "noflip"
    reference = golden[f"{image_name}/keypoints_{key}"]
    assert keypoints.shape == reference.shape
    xy_diff = np.abs(keypoints[..., :2] - reference[..., :2]).max()
    score_diff = np.abs(keypoints[..., 2] - reference[..., 2]).max()
    assert xy_diff <= XY_TOLERANCE, f"coordinate diff {xy_diff} > {XY_TOLERANCE}"
    assert score_diff <= SCORE_TOLERANCE, f"score diff {score_diff} > {SCORE_TOLERANCE}"


@pytest.mark.parametrize("flip_test", [True, False])
def test_heatmap_parity(model, golden, flip_test):
    image = cv2.imread(str(IMAGE_PATHS["crop"]))
    boxes = golden["crop/boxes"]
    _, heatmaps = model.detect(image, boxes, flip_test=flip_test, return_heatmap=True)

    key = "flip" if flip_test else "noflip"
    reference = golden[f"crop/heatmap_{key}"]
    assert heatmaps.shape == reference.shape
    diff = np.abs(heatmaps - reference).max()
    assert diff <= HEATMAP_TOLERANCE, f"heatmap diff {diff} > {HEATMAP_TOLERANCE}"


def test_empty_boxes(model):
    image = cv2.imread(str(IMAGE_PATHS["crop"]))
    keypoints = model.detect(image, np.zeros((0, 5), dtype=np.float32))
    assert keypoints.shape == (0, 28, 3)
