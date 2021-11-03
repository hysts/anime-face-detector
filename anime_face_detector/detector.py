from __future__ import annotations

import pathlib
import warnings
from typing import Optional, Union

import cv2
import mmcv
import numpy as np
import torch.nn as nn
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo


class LandmarkDetector:
    def __init__(
            self,
            landmark_detector_config_or_path: Union[mmcv.Config, str,
                                                    pathlib.Path],
            landmark_detector_checkpoint_path: Union[str, pathlib.Path],
            face_detector_config_or_path: Optional[Union[mmcv.Config, str,
                                                         pathlib.Path]] = None,
            face_detector_checkpoint_path: Optional[Union[
                str, pathlib.Path]] = None,
            device: str = 'cuda:0',
            flip_test: bool = True,
            box_scale_factor: float = 1.1):
        landmark_config = self._load_config(landmark_detector_config_or_path)
        self.dataset_info = DatasetInfo(
            landmark_config.dataset_info)  # type: ignore
        face_detector_config = self._load_config(face_detector_config_or_path)

        self.landmark_detector = self._init_pose_model(
            landmark_config, landmark_detector_checkpoint_path, device,
            flip_test)
        self.face_detector = self._init_face_detector(
            face_detector_config, face_detector_checkpoint_path, device)

        self.box_scale_factor = box_scale_factor

    @staticmethod
    def _load_config(
        config_or_path: Optional[Union[mmcv.Config, str, pathlib.Path]]
    ) -> Optional[mmcv.Config]:
        if config_or_path is None or isinstance(config_or_path, mmcv.Config):
            return config_or_path
        return mmcv.Config.fromfile(config_or_path)

    @staticmethod
    def _init_pose_model(config: mmcv.Config,
                         checkpoint_path: Union[str, pathlib.Path],
                         device: str, flip_test: bool) -> nn.Module:
        if isinstance(checkpoint_path, pathlib.Path):
            checkpoint_path = checkpoint_path.as_posix()
        model = init_pose_model(config, checkpoint_path, device=device)
        model.cfg.model.test_cfg.flip_test = flip_test
        return model

    @staticmethod
    def _init_face_detector(config: Optional[mmcv.Config],
                            checkpoint_path: Optional[Union[str,
                                                            pathlib.Path]],
                            device: str) -> Optional[nn.Module]:
        if config is not None:
            if isinstance(checkpoint_path, pathlib.Path):
                checkpoint_path = checkpoint_path.as_posix()
            model = init_detector(config, checkpoint_path, device=device)
        else:
            model = None
        return model

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        # predicted boxes using mmdet model have the format of
        # [x0, y0, x1, y1, score]
        boxes = inference_detector(self.face_detector, image)[0]
        # scale boxes by `self.box_scale_factor`
        boxes = self._update_pred_box(boxes)
        return boxes

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

    def _detect_landmarks(
            self, image: np.ndarray,
            boxes: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        preds, _ = inference_top_down_pose_model(
            self.landmark_detector,
            image,
            boxes,
            format='xyxy',
            dataset_info=self.dataset_info,
            return_heatmap=False)
        return preds

    @staticmethod
    def _load_image(
            image_or_path: Union[np.ndarray, str, pathlib.Path]) -> np.ndarray:
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        elif isinstance(image_or_path, pathlib.Path):
            image = cv2.imread(image_or_path.as_posix())
        else:
            raise ValueError
        return image

    def __call__(
        self,
        image_or_path: Union[np.ndarray, str, pathlib.Path],
        boxes: Optional[list[np.ndarray]] = None
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
                    'Neither the face detector nor the bounding box is '
                    'specified. So the entire image is treated as the face '
                    'region.')
                h, w = image.shape[:2]
                boxes = [np.array([0, 0, w - 1, h - 1, 1])]
        box_list = [{'bbox': box} for box in boxes]
        return self._detect_landmarks(image, box_list)
