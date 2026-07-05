"""Anime face detectors (Faster R-CNN and YOLOv3), pure PyTorch.

This reimplements what the original mm*-based stack did at inference
time: mmdet's test pipeline (keep-ratio resize, normalization, padding
to a stride-32 divisor) and the single-image test paths of the
FasterRCNN and YOLOV3 detectors. The hyperparameters are those of the
original ``configs/mmdet/*.py`` (see git history), with which the
released checkpoints were trained.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from ._vendor.mmdet import (
    FPN,
    AnchorGenerator,
    Darknet,
    DeltaXYWHBBoxCoder,
    ResNet,
    RPNHead,
    Shared2FCBBoxHead,
    SingleRoIExtractor,
    StandardRoIHead,
    YOLOAnchorGenerator,
    YOLOBBoxCoder,
    YOLOV3Head,
    YOLOV3Neck,
)

FACE_DETECTOR_NAMES = ("faster-rcnn", "yolov3")
PAD_SIZE_DIVISOR = 32


def _rescale_size(width: int, height: int, max_long_edge: int, max_short_edge: int) -> tuple[int, int]:
    """Compute the keep-ratio target size, like mmcv.rescale_size."""
    scale_factor = min(max_long_edge / max(height, width), max_short_edge / min(height, width))
    return int(width * scale_factor + 0.5), int(height * scale_factor + 0.5)


def _preprocess(
    image: np.ndarray,
    img_scale: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[torch.Tensor, dict]:
    """Replicate mmdet's test pipeline.

    Keep-ratio resize, BGR to RGB, normalize, pad to a stride-32 divisor.

    Args:
        image: Input image with BGR channel order, shaped (H, W, 3).
        img_scale: (max_long_edge, max_short_edge) of the resize.
        mean: Per-channel (RGB) normalization mean.
        std: Per-channel (RGB) normalization std.

    Returns:
        The (1, 3, H, W) input tensor and the mmdet-style img_meta dict.
    """
    height, width = image.shape[:2]
    new_width, new_height = _rescale_size(width, height, *img_scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    scale_factor = np.array(
        [new_width / width, new_height / height, new_width / width, new_height / height],
        dtype=np.float32,
    )

    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

    pad_height = (new_height + PAD_SIZE_DIVISOR - 1) // PAD_SIZE_DIVISOR * PAD_SIZE_DIVISOR
    pad_width = (new_width + PAD_SIZE_DIVISOR - 1) // PAD_SIZE_DIVISOR * PAD_SIZE_DIVISOR
    padded = np.zeros((pad_height, pad_width, 3), dtype=np.float32)
    padded[:new_height, :new_width] = img

    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    img_meta = {
        "img_shape": (new_height, new_width, 3),
        "ori_shape": (height, width, 3),
        "pad_shape": (pad_height, pad_width, 3),
        "scale_factor": scale_factor,
    }
    return tensor, img_meta


class FasterRCNNFaceDetector(nn.Module):
    """The original anime-face Faster R-CNN (R50-FPN), inference only.

    Submodule names match the mmdet ``FasterRCNN`` model, so the released
    checkpoint loads with ``strict=True``.
    """

    IMG_SCALE = (1333, 800)
    MEAN = (123.675, 116.28, 103.53)
    STD = (58.395, 57.12, 57.375)

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet(depth=50, num_stages=4, out_indices=(0, 1, 2, 3))
        self.neck = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
        self.rpn_head = RPNHead(
            in_channels=256,
            feat_channels=256,
            anchor_generator=AnchorGenerator(scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
            bbox_coder=DeltaXYWHBBoxCoder(target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)),
            test_cfg={
                "nms_pre": 1000,
                "max_per_img": 1000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
        )
        self.roi_head = StandardRoIHead(
            bbox_roi_extractor=SingleRoIExtractor(
                output_size=7, sampling_ratio=0, out_channels=256, featmap_strides=[4, 8, 16, 32]
            ),
            bbox_head=Shared2FCBBoxHead(
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=DeltaXYWHBBoxCoder(target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(0.1, 0.1, 0.2, 0.2)),
                reg_class_agnostic=False,
            ),
            test_cfg={
                "score_thr": 0.05,
                "nms": {"type": "nms", "iou_threshold": 0.5},
                "max_per_img": 100,
            },
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect faces in a BGR image.

        Returns (N, 5) boxes as [x0, y0, x1, y1, score] in the original
        image coordinates.
        """
        img, img_meta = _preprocess(image, self.IMG_SCALE, self.MEAN, self.STD)
        feats = self.neck(self.backbone(img.to(self.device)))
        proposal_list = self.rpn_head.simple_test_rpn(feats, [img_meta])
        det_bboxes, _ = self.roi_head.simple_test_bboxes(
            feats, [img_meta], proposal_list, self.roi_head.test_cfg, rescale=True
        )
        return det_bboxes[0].cpu().numpy()


class YOLOv3FaceDetector(nn.Module):
    """The original anime-face YOLOv3 (Darknet-53), inference only.

    Submodule names match the mmdet ``YOLOV3`` model, so the released
    checkpoint loads with ``strict=True``.
    """

    IMG_SCALE = (608, 608)
    MEAN = (0.0, 0.0, 0.0)
    STD = (255.0, 255.0, 255.0)

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Darknet(depth=53, out_indices=(3, 4, 5))
        self.neck = YOLOV3Neck(num_scales=3, in_channels=[1024, 512, 256], out_channels=[512, 256, 128])
        self.bbox_head = YOLOV3Head(
            num_classes=1,
            in_channels=[512, 256, 128],
            out_channels=[1024, 512, 256],
            anchor_generator=YOLOAnchorGenerator(
                base_sizes=[
                    [(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)],
                ],
                strides=[32, 16, 8],
            ),
            bbox_coder=YOLOBBoxCoder(),
            featmap_strides=[32, 16, 8],
            test_cfg={
                "nms_pre": 1000,
                "min_bbox_size": 0,
                "score_thr": 0.05,
                "conf_thr": 0.005,
                "nms": {"type": "nms", "iou_threshold": 0.45},
                "max_per_img": 100,
            },
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect faces in a BGR image.

        Returns (N, 5) boxes as [x0, y0, x1, y1, score] in the original
        image coordinates.
        """
        img, img_meta = _preprocess(image, self.IMG_SCALE, self.MEAN, self.STD)
        feats = self.neck(self.backbone(img.to(self.device)))
        pred_maps = self.bbox_head(feats)[0]
        results = self.bbox_head.get_bboxes(pred_maps, [img_meta], rescale=True)
        return results[0][0].cpu().numpy()


def load_face_detector(
    model_name: str, checkpoint_path: str, device: str = "cpu"
) -> FasterRCNNFaceDetector | YOLOv3FaceDetector:
    """Build a face detector and load a checkpoint with strict keys."""
    if model_name == "faster-rcnn":
        model = FasterRCNNFaceDetector()
    elif model_name == "yolov3":
        model = YOLOv3FaceDetector()
    else:
        msg = f"model_name must be one of {FACE_DETECTOR_NAMES}, got {model_name!r}"
        raise ValueError(msg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
