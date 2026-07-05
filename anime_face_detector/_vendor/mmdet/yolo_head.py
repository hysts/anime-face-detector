# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
#
# Vendored from mmdetection v2.28.2, mmdet/models/dense_heads/yolo_head.py
# (Apache License 2.0, see ../LICENSE.mmdetection). Modifications:
# removed the registry decorator, losses, weight init and all
# training/ONNX/augmented-test code (kept forward and get_bboxes);
# ConvModule comes from the local mmcv shim with the BN + LeakyReLU(0.1)
# configuration hardcoded; the anchor generator and bbox coder are
# passed in directly instead of built from registry config dicts;
# test_cfg is a plain dict. Submodule attribute names are unchanged, so
# state_dict keys match the original.
import numpy as np
import torch
import torch.nn as nn

from ..mmcv import ConvModule
from .anchor_generator import YOLOAnchorGenerator
from .bbox_coders import YOLOBBoxCoder
from .bbox_nms import multiclass_nms

NORM_CFG = dict(type='BN')
ACT_CFG = dict(type='LeakyReLU', negative_slope=0.1)


class YOLOV3Head(nn.Module):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (YOLOAnchorGenerator): The anchor generator.
        bbox_coder (YOLOBBoxCoder): The bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        test_cfg (dict): Testing config of YOLOV3 head, e.g.
            dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05,
            conf_thr=0.005, nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=None,
                 bbox_coder=None,
                 featmap_strides=[32, 16, 8],
                 test_cfg=None):
        super().__init__()
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.test_cfg = test_cfg

        assert isinstance(bbox_coder, YOLOBBoxCoder)
        assert isinstance(anchor_generator, YOLOAnchorGenerator)
        self.bbox_coder = bbox_coder
        self.prior_generator = anchor_generator

        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(
            self.prior_generator.num_base_priors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                norm_cfg=NORM_CFG,
                act_cfg=ACT_CFG)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_base_priors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    def get_bboxes(self, pred_maps, img_metas, cfg=None, rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (dict | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        num_imgs = len(img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))

        if flatten_objectness.size(0) == 0:
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            flatten_bboxes /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1],
                                           1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)

        det_results = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg['score_thr'],
                cfg['nms'],
                cfg['max_per_img'],
                score_factors=objectness)
            det_results.append(tuple([det_bboxes, det_labels]))
        return det_results
