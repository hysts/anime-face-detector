"""Anime face detectors (Faster R-CNN and YOLOv3), pure PyTorch."""

from __future__ import annotations

FACE_DETECTOR_NAMES = ("faster-rcnn", "yolov3")


# Lint exemptions: placeholder signature until the ported detectors land.
def load_face_detector(model_name: str, checkpoint_path: str, device: str = "cpu"):  # noqa: ANN202, ARG001
    """Build a face detector and load a checkpoint with strict keys."""
    if model_name not in FACE_DETECTOR_NAMES:
        msg = f"model_name must be one of {FACE_DETECTOR_NAMES}, got {model_name!r}"
        raise ValueError(msg)
    msg = "The pure PyTorch face detectors are not implemented yet."
    raise NotImplementedError(msg)
