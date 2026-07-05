"""Convert a mm*-era .pth checkpoint to the safetensors file this package loads.

The released checkpoints (and self-trained ones from the mmdet/mmpose era)
are pickle files that wrap the model state dict together with training
metadata. This package loads safetensors only, so such checkpoints must go
through this one-time conversion.
"""

import argparse
import hashlib
import logging
import pathlib

import torch
from safetensors.torch import save_file

from ._face import FasterRCNNFaceDetector, YOLOv3FaceDetector
from ._landmark import LandmarkModel

logger = logging.getLogger(__name__)

MODEL_NAMES = ("faster-rcnn", "yolov3", "hrnetv2")
_SOURCE_STACK = {
    "faster-rcnn": "mmdetection v2.28.2",
    "yolov3": "mmdetection v2.28.2",
    "hrnetv2": "mmpose v0.29.0",
}


def _reference_model(model_name: str) -> torch.nn.Module:
    if model_name == "hrnetv2":
        return LandmarkModel()
    if model_name == "faster-rcnn":
        return FasterRCNNFaceDetector()
    return YOLOv3FaceDetector()


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Return the model state dict from a loaded checkpoint object.

    Accepts both the mm*-style ``{"meta": ..., "state_dict": ...}`` layout
    and a bare state dict.
    """
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        error_msg = f"The checkpoint must be a dict, but got {type(checkpoint)}."
        raise TypeError(error_msg)
    if not state_dict:
        raise ValueError("The checkpoint contains an empty state dict.")
    non_tensors = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
    if non_tensors:
        error_msg = f"The checkpoint is not a state dict: non-tensor values at {non_tensors}."
        raise TypeError(error_msg)
    return state_dict


def validate_state_dict(state_dict: dict[str, torch.Tensor], model_name: str) -> None:
    """Check the state dict keys and shapes against the given architecture."""
    reference = _reference_model(model_name).state_dict()
    missing = sorted(reference.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - reference.keys())
    if missing or unexpected:
        error_msg = (
            f"The state dict does not match the {model_name} architecture. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
        raise ValueError(error_msg)
    mismatched = [
        f"{key}: {tuple(state_dict[key].shape)} != {tuple(reference[key].shape)}"
        for key in reference
        if state_dict[key].shape != reference[key].shape
    ]
    if mismatched:
        error_msg = f"The state dict does not match the {model_name} architecture. Shape mismatches: {mismatched}."
        raise ValueError(error_msg)


def convert(input_path: pathlib.Path, output_path: pathlib.Path, model_name: str | None = None) -> None:
    """Convert a .pth checkpoint to a .safetensors file."""
    if output_path.suffix != ".safetensors":
        error_msg = f"The output file must have a .safetensors extension, but got {output_path.name}."
        raise ValueError(error_msg)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(checkpoint)
    metadata = {
        "converted_by": "anime-face-detector-convert",
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    if model_name is not None:
        validate_state_dict(state_dict, model_name)
        metadata["model"] = model_name
        metadata["source_stack"] = _SOURCE_STACK[model_name]
    state_dict = {key: value.contiguous() for key, value in state_dict.items()}
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_file(state_dict, output_path, metadata=metadata)
    logger.info("Saved %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="Input .pth checkpoint.")
    parser.add_argument("output", type=pathlib.Path, help="Output .safetensors file.")
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_NAMES,
        help="If specified, the state dict is validated against this "
        "architecture, and the model name is recorded in the file metadata.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(args.input, args.output, args.model)
