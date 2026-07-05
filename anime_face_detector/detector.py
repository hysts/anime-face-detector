from __future__ import annotations

import pathlib
import warnings

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ._face import load_face_detector
from ._landmark import load_landmark_model

WEIGHT_BASE_URL = "https://github.com/hysts/anime-face-detector/releases/download/v0.0.1"
MODEL_NAMES = ("faster-rcnn", "yolov3", "hrnetv2")
HF_MODEL_REPOS = {
    "faster-rcnn": "hysts/anime-face-detector-faster-rcnn",
    "yolov3": "hysts/anime-face-detector-yolov3",
    "hrnetv2": "hysts/anime-face-detector-hrnetv2",
}


def get_checkpoint_path(model_name: str) -> pathlib.Path:
    """Return the local path of a pretrained safetensors checkpoint.

    The file is downloaded from the Hugging Face Hub into the Hub cache
    if needed.
    """
    if model_name not in MODEL_NAMES:
        msg = f"model_name must be one of {MODEL_NAMES}, got {model_name!r}"
        raise ValueError(msg)
    return pathlib.Path(hf_hub_download(HF_MODEL_REPOS[model_name], "model.safetensors"))


def get_legacy_checkpoint_path(model_name: str) -> pathlib.Path:
    """Return the local path of an original mm*-era .pth checkpoint.

    The file is downloaded from the GitHub release into the torch hub
    cache if needed. These are the checkpoints
    ``anime-face-detector-convert`` takes as input; ``create_detector``
    itself uses the safetensors files from the Hugging Face Hub.
    """
    if model_name not in MODEL_NAMES:
        msg = f"model_name must be one of {MODEL_NAMES}, got {model_name!r}"
        raise ValueError(msg)
    if model_name in ["faster-rcnn", "yolov3"]:
        file_name = f"mmdet_anime-face_{model_name}.pth"
    else:
        file_name = f"mmpose_anime-face_{model_name}.pth"

    model_dir = pathlib.Path(torch.hub.get_dir()) / "checkpoints"
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / file_name
    if not model_path.exists():
        torch.hub.download_url_to_file(f"{WEIGHT_BASE_URL}/{file_name}", model_path.as_posix())

    return model_path


class LandmarkDetector:
    """Detect anime faces and their facial landmarks.

    Args:
        landmark_checkpoint_path: Path to the HRNetV2 landmark checkpoint.
        face_detector_name: 'yolov3', 'faster-rcnn', or None to skip face
            detection (bounding boxes must then be given at call time).
        face_detector_checkpoint_path: Path to the face detector checkpoint.
        device: Device specifier, e.g. 'cuda:0' or 'cpu'.
        flip_test: Whether to average landmark predictions over the
            horizontally flipped input.
        box_scale_factor: Scale factor applied to detected face boxes
            before landmark estimation.
    """

    def __init__(
        self,
        landmark_checkpoint_path: str | pathlib.Path,
        face_detector_name: str | None = None,
        face_detector_checkpoint_path: str | pathlib.Path | None = None,
        device: str = "cuda:0",
        flip_test: bool = True,
        box_scale_factor: float = 1.1,
    ) -> None:
        self.landmark_detector = load_landmark_model(str(landmark_checkpoint_path), device=device)
        if face_detector_name is not None:
            if face_detector_checkpoint_path is None:
                raise ValueError("face_detector_checkpoint_path is required when face_detector_name is set.")
            self.face_detector = load_face_detector(
                face_detector_name, str(face_detector_checkpoint_path), device=device
            )
        else:
            self.face_detector = None
        self.flip_test = flip_test
        self.box_scale_factor = box_scale_factor

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        # predicted boxes have the format of [x0, y0, x1, y1, score]
        boxes = self.face_detector.detect(image)
        # scale boxes by `self.box_scale_factor`
        return self._update_pred_box(boxes)

    def _update_pred_box(self, pred_boxes: np.ndarray) -> list[np.ndarray]:
        boxes = []
        for pred_box in pred_boxes:
            box = pred_box[:4]
            size = box[2:] - box[:2] + 1
            new_size = size * self.box_scale_factor
            center = (box[:2] + box[2:]) / 2
            tl = center - new_size / 2
            br = tl + new_size
            pred_box[:4] = np.concatenate([tl, br])
            boxes.append(pred_box)
        return boxes

    @staticmethod
    def _load_image(image_or_path: np.ndarray | str | pathlib.Path) -> np.ndarray:
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        elif isinstance(image_or_path, pathlib.Path):
            image = cv2.imread(image_or_path.as_posix())
        else:
            msg = f"Unsupported image type: {type(image_or_path)}"
            raise TypeError(msg)
        return image

    def __call__(
        self,
        image_or_path: np.ndarray | str | pathlib.Path,
        boxes: list[np.ndarray] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Detect face landmarks.

        Args:
            image_or_path: An image with BGR channel order or an image path.
            boxes: A list of bounding boxes for faces. Each bounding box
                should be of the form [x0, y0, x1, y1, [score]].

        Returns: A list of detection results. Each detection result has
            bounding box of the form [x0, y0, x1, y1, [score]], and landmarks
            of the form [x, y, score].
        """
        image = self._load_image(image_or_path)
        if boxes is None:
            if self.face_detector is not None:
                boxes = self._detect_faces(image)
            else:
                warnings.warn(
                    "Neither the face detector nor the bounding box is "
                    "specified. So the entire image is treated as the face "
                    "region.",
                    stacklevel=2,
                )
                h, w = image.shape[:2]
                boxes = [np.array([0, 0, w - 1, h - 1, 1])]
        if len(boxes) == 0:
            return []
        box_array = np.stack([np.asarray(box, dtype=np.float32) for box in boxes])
        keypoints = self.landmark_detector.detect(image, box_array, flip_test=self.flip_test)
        return [
            {"bbox": np.asarray(box, dtype=np.float32), "keypoints": kpts}
            for box, kpts in zip(boxes, keypoints, strict=True)
        ]


def create_detector(
    face_detector_name: str = "yolov3",
    landmark_model_name: str = "hrnetv2",
    device: str = "cuda:0",
    flip_test: bool = True,
    box_scale_factor: float = 1.1,
) -> LandmarkDetector:
    """Create a LandmarkDetector with pretrained weights.

    Args:
        face_detector_name: 'yolov3' or 'faster-rcnn'.
        landmark_model_name: 'hrnetv2'.
        device: Device specifier, e.g. 'cuda:0' or 'cpu'.
        flip_test: Whether to average landmark predictions over the
            horizontally flipped input.
        box_scale_factor: Scale factor applied to detected face boxes
            before landmark estimation.
    """
    if face_detector_name not in ["yolov3", "faster-rcnn"]:
        msg = f"face_detector_name must be 'yolov3' or 'faster-rcnn', got {face_detector_name!r}"
        raise ValueError(msg)
    if landmark_model_name != "hrnetv2":
        msg = f"landmark_model_name must be 'hrnetv2', got {landmark_model_name!r}"
        raise ValueError(msg)
    return LandmarkDetector(
        get_checkpoint_path(landmark_model_name),
        face_detector_name=face_detector_name,
        face_detector_checkpoint_path=get_checkpoint_path(face_detector_name),
        device=device,
        flip_test=flip_test,
        box_scale_factor=box_scale_factor,
    )
