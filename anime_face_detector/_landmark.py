"""HRNetV2 facial landmark model and its top-down inference pipeline.

This reimplements, in plain PyTorch, exactly what the original mm*-based
stack did at inference time: mmpose's ``inference_top_down_pose_model``
preprocessing, the ``TopDown`` test forward with flip test, and unbiased
(DARK) heatmap decoding. The hyperparameters are those of the original
``configs/mmpose/hrnetv2.py`` (see git history), with which the released
checkpoint was trained.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from ._vendor.mmpose import (
    HRNet,
    TopdownHeatmapSimpleHead,
    bbox_xywh2cs,
    bbox_xyxy2xywh,
    get_affine_transform,
    keypoints_from_heatmaps,
)

IMAGE_SIZE = (256, 256)
# Bbox padding factor applied when converting boxes to center/scale
# (mmpose's TopDownGetBboxCenterScale default).
BBOX_PADDING = 1.25
# ImageNet statistics, matching NormalizeTensor in the original config.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
# From the `swap` fields of the original config's dataset_info.
FLIP_PAIRS = (
    (0, 4),
    (1, 3),
    (5, 10),
    (6, 9),
    (7, 8),
    (11, 19),
    (12, 18),
    (13, 17),
    (14, 22),
    (15, 21),
    (16, 20),
    (24, 26),
)

HRNET_EXTRA = {
    "stage1": {
        "num_modules": 1,
        "num_branches": 1,
        "block": "BOTTLENECK",
        "num_blocks": (4,),
        "num_channels": (64,),
    },
    "stage2": {
        "num_modules": 1,
        "num_branches": 2,
        "block": "BASIC",
        "num_blocks": (4, 4),
        "num_channels": (18, 36),
    },
    "stage3": {
        "num_modules": 4,
        "num_branches": 3,
        "block": "BASIC",
        "num_blocks": (4, 4, 4),
        "num_channels": (18, 36, 72),
    },
    "stage4": {
        "num_modules": 3,
        "num_branches": 4,
        "block": "BASIC",
        "num_blocks": (4, 4, 4, 4),
        "num_channels": (18, 36, 72, 144),
        "multiscale_output": True,
    },
    "upsample": {"mode": "bilinear", "align_corners": False},
}

HEAD_TEST_CFG = {
    "post_process": "unbiased",
    "shift_heatmap": True,
    "modulate_kernel": 11,
}


class LandmarkModel(nn.Module):
    """HRNetV2 with a heatmap head, matching the released checkpoint.

    The submodule names (``backbone``, ``keypoint_head``) match the
    original mmpose ``TopDown`` model, so the checkpoint's state_dict
    loads with ``strict=True``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = HRNet(HRNET_EXTRA)
        self.keypoint_head = TopdownHeatmapSimpleHead(
            in_channels=[18, 36, 72, 144],
            out_channels=28,
            num_deconv_layers=0,
            extra={"final_conv_kernel": 1, "num_conv_layers": 1, "num_conv_kernels": (1,)},
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            test_cfg=HEAD_TEST_CFG,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _heatmaps(self, img: torch.Tensor, flip_test: bool) -> np.ndarray:
        features = self.backbone(img)
        output_heatmap = self.keypoint_head.inference_model(features, flip_pairs=None)

        if flip_test:
            img_flipped = img.flip(3)
            features_flipped = self.backbone(img_flipped)
            output_flipped_heatmap = self.keypoint_head.inference_model(features_flipped, list(FLIP_PAIRS))
            output_heatmap = (output_heatmap + output_flipped_heatmap) / 2

        return output_heatmap

    @torch.inference_mode()
    def detect(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        flip_test: bool = True,
        return_heatmap: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Estimate landmarks for each face box in an image.

        Args:
            image: Input image with BGR channel order, shaped (H, W, 3).
            boxes: Face bounding boxes shaped (N, 4) or (N, 5) in
                (x0, y0, x1, y1, [score]) format.
            flip_test: Whether to average predictions over the
                horizontally flipped input.
            return_heatmap: Whether to also return the raw output
                heatmaps shaped (N, 28, 64, 64).

        Returns:
            Landmarks shaped (N, 28, 3) as (x, y, score), and the raw
            heatmaps if ``return_heatmap`` is set.
        """
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis]
        if len(boxes) == 0:
            keypoints = np.zeros((0, 28, 3), dtype=np.float32)
            if return_heatmap:
                return keypoints, np.zeros((0, 28, 64, 64), dtype=np.float32)
            return keypoints

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aspect_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]

        centers = np.zeros((len(boxes), 2), dtype=np.float32)
        scales = np.zeros((len(boxes), 2), dtype=np.float32)
        crops = []
        for i, box in enumerate(bbox_xyxy2xywh(boxes)):
            center, scale = bbox_xywh2cs(box, aspect_ratio=aspect_ratio, padding=BBOX_PADDING)
            centers[i] = center
            scales[i] = scale
            trans = get_affine_transform(center, scale, 0, IMAGE_SIZE)
            crops.append(cv2.warpAffine(image_rgb, trans, IMAGE_SIZE, flags=cv2.INTER_LINEAR))

        batch = torch.from_numpy(np.stack(crops).astype(np.float32)).permute(0, 3, 1, 2).div_(255.0)
        mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1)
        batch.sub_(mean).div_(std)

        heatmaps = self._heatmaps(batch.to(self.device), flip_test)
        preds, maxvals = keypoints_from_heatmaps(
            heatmaps,
            centers,
            scales,
            post_process=HEAD_TEST_CFG["post_process"],
            kernel=HEAD_TEST_CFG["modulate_kernel"],
        )
        keypoints = np.concatenate([preds, maxvals], axis=-1).astype(np.float32)
        if return_heatmap:
            return keypoints, heatmaps
        return keypoints


def load_landmark_model(checkpoint_path: str, device: str = "cpu") -> LandmarkModel:
    """Build the landmark model and load a checkpoint with strict keys."""
    model = LandmarkModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
