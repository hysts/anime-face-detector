"""Face detector parity against the golden outputs of the mm*-based stack.

Boxes with score >= 0.5 must match the goldens in count, coordinates
(within 1 px) and score (within 1e-2, see test_landmark_parity.py for
why the score tolerance is looser than originally planned). The full
end-to-end pipeline (detector + landmarks through the public API) is
checked the same way.
"""

import pathlib

import cv2
import numpy as np
import pytest

from anime_face_detector import LandmarkDetector, get_legacy_checkpoint_path

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

XY_TOLERANCE = 1.0
SCORE_TOLERANCE = 1e-2
KEYPOINT_XY_TOLERANCE = 0.5
# On the crowd image with dozens of small faces the torch-version numeric
# drift is a bit larger: heatmap peak values shift by up to ~2e-2, and on
# near-flat peaks the argmax cell can move to a neighbor, moving an
# isolated keypoint by up to ~0.7 px (all others stay within ~0.03 px).
CROWD_KEYPOINT_XY_TOLERANCE = 1.0
CROWD_KEYPOINT_SCORE_TOLERANCE = 3e-2
CONFIDENT_SCORE = 0.5

IMAGE_PATHS = {
    "input": REPO_ROOT / "assets" / "input.jpg",
    "small": GOLDEN_DIR / "images" / "small.png",
    "crop": GOLDEN_DIR / "images" / "crop.png",
    "noface": GOLDEN_DIR / "images" / "noface.png",
}


@pytest.fixture(scope="session", params=["faster-rcnn", "yolov3"])
def detector_name(request):
    return request.param


@pytest.fixture(scope="session")
def detector(detector_name):
    return LandmarkDetector(
        get_legacy_checkpoint_path("hrnetv2"),
        face_detector_name=detector_name,
        face_detector_checkpoint_path=get_legacy_checkpoint_path(detector_name),
        device="cpu",
    )


def _confident(boxes: np.ndarray) -> np.ndarray:
    return boxes[boxes[:, 4] >= CONFIDENT_SCORE]


@pytest.mark.parametrize("image_name", ["input", "small", "crop", "noface"])
def test_detector_parity(detector, detector_name, image_name):
    image = cv2.imread(str(IMAGE_PATHS[image_name]))
    with np.load(GOLDEN_DIR / f"detector_{detector_name}.npz") as data:
        reference = data[f"{image_name}/raw_boxes"]

    boxes = detector.face_detector.detect(image)
    boxes, reference = _confident(boxes), _confident(reference)
    assert len(boxes) == len(reference)
    if len(boxes) > 0:
        xy_diff = np.abs(boxes[:, :4] - reference[:, :4]).max()
        score_diff = np.abs(boxes[:, 4] - reference[:, 4]).max()
        assert xy_diff <= XY_TOLERANCE, f"box diff {xy_diff} > {XY_TOLERANCE}"
        assert score_diff <= SCORE_TOLERANCE, f"score diff {score_diff} > {SCORE_TOLERANCE}"


@pytest.mark.parametrize("image_name", ["crop", "noface"])
@pytest.mark.parametrize("flip_test", [True, False])
def test_e2e_parity(detector, detector_name, image_name, flip_test):
    image = cv2.imread(str(IMAGE_PATHS[image_name]))
    key = "flip" if flip_test else "noflip"
    with np.load(GOLDEN_DIR / f"e2e_{detector_name}.npz") as data:
        ref_bboxes = data[f"{image_name}/bboxes_{key}"]
        ref_keypoints = data[f"{image_name}/keypoints_{key}"]

    detector.flip_test = flip_test
    preds = detector(image)
    bboxes = np.stack([p["bbox"] for p in preds]) if preds else np.zeros((0, 5), np.float32)
    keypoints = np.stack([p["keypoints"] for p in preds]) if preds else np.zeros((0, 28, 3), np.float32)

    mask = bboxes[:, 4] >= CONFIDENT_SCORE if len(bboxes) else np.zeros(0, dtype=bool)
    ref_mask = ref_bboxes[:, 4] >= CONFIDENT_SCORE if len(ref_bboxes) else np.zeros(0, dtype=bool)
    bboxes, keypoints = bboxes[mask], keypoints[mask]
    ref_bboxes, ref_keypoints = ref_bboxes[ref_mask], ref_keypoints[ref_mask]

    assert len(bboxes) == len(ref_bboxes)
    if len(bboxes) > 0:
        assert np.abs(bboxes[:, :4] - ref_bboxes[:, :4]).max() <= XY_TOLERANCE
        kpt_xy_diff = np.abs(keypoints[..., :2] - ref_keypoints[..., :2]).max()
        kpt_score_diff = np.abs(keypoints[..., 2] - ref_keypoints[..., 2]).max()
        assert kpt_xy_diff <= KEYPOINT_XY_TOLERANCE, f"keypoint diff {kpt_xy_diff}"
        assert kpt_score_diff <= SCORE_TOLERANCE, f"keypoint score diff {kpt_score_diff}"


@pytest.mark.parametrize("flip_test", [True, False])
def test_e2e_parity_many_faces(detector, detector_name, flip_test):
    """The full crowd image: dozens of faces through the whole pipeline."""
    image = cv2.imread(str(IMAGE_PATHS["input"]))
    key = "flip" if flip_test else "noflip"
    with np.load(GOLDEN_DIR / f"e2e_{detector_name}.npz") as data:
        ref_bboxes = data[f"input/bboxes_{key}"]
        ref_keypoints = data[f"input/keypoints_{key}"]

    detector.flip_test = flip_test
    preds = detector(image)
    bboxes = np.stack([p["bbox"] for p in preds])
    keypoints = np.stack([p["keypoints"] for p in preds])

    mask = bboxes[:, 4] >= CONFIDENT_SCORE
    ref_mask = ref_bboxes[:, 4] >= CONFIDENT_SCORE
    bboxes, keypoints = bboxes[mask], keypoints[mask]
    ref_bboxes, ref_keypoints = ref_bboxes[ref_mask], ref_keypoints[ref_mask]

    assert len(bboxes) == len(ref_bboxes)
    assert np.abs(bboxes[:, :4] - ref_bboxes[:, :4]).max() <= XY_TOLERANCE
    assert np.abs(keypoints[..., :2] - ref_keypoints[..., :2]).max() <= CROWD_KEYPOINT_XY_TOLERANCE
    assert np.abs(keypoints[..., 2] - ref_keypoints[..., 2]).max() <= CROWD_KEYPOINT_SCORE_TOLERANCE
