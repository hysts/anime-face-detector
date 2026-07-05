"""Checkpoint loading helpers."""

from __future__ import annotations

import pathlib

import torch
from safetensors.torch import load_file


def load_state_dict_from_path(checkpoint_path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    """Load a model state dict from a .safetensors or mm*-era .pth file."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return load_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)
