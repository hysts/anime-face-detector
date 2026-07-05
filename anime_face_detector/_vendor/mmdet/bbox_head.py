# Copyright (c) OpenMMLab. All rights reserved.
#
# Vendored from mmdetection v2.28.2,
# mmdet/models/roi_heads/bbox_heads/bbox_head.py and
# mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py (Apache License
# 2.0, see ../LICENSE.mmdetection), merged into one module.
# Modifications: removed the registry decorators, losses, weight init
# and all training/ONNX code; kept only the options exercised by the
# Shared2FCBBoxHead configuration (with_avg_pool=False, Linear
# predictors, softmax classification); the bbox coder is passed in
# directly instead of built from a config dict; test cfg values are
# plain dict entries. Submodule attribute names are unchanged, so
# state_dict keys match the original.
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .bbox_nms import multiclass_nms


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=None,
                 reg_class_agnostic=False):
        super().__init__()
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic

        self.bbox_coder = bbox_coder

    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (dict): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg['score_thr'],
                                                    cfg['nms'],
                                                    cfg['max_per_img'])

            return det_bboxes, det_labels


class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared fc layers and two optional
    separated branches. Only the fully-connected parts used by
    Shared2FCBBoxHead are kept in this vendored copy.

    .. code-block:: none

                          /-> cls fcs -> cls
        shared fcs
                          \-> reg fcs -> reg
    """

    def __init__(self,
                 num_shared_fcs=0,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert num_shared_fcs > 0
        self.num_shared_fcs = num_shared_fcs
        self.fc_out_channels = fc_out_channels

        # add shared fcs
        self.shared_fcs, last_layer_dim = self._add_fc_branch(
            self.num_shared_fcs, self.in_channels)
        self.shared_out_channels = last_layer_dim
        self.cls_last_dim = last_layer_dim
        self.reg_last_dim = last_layer_dim

        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(
            in_features=self.cls_last_dim,
            out_features=self.num_classes + 1)
        out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
        self.fc_reg = nn.Linear(
            in_features=self.reg_last_dim, out_features=out_dim_reg)

    def _add_fc_branch(self, num_branch_fcs, in_channels):
        """Add shared fc branch."""
        last_layer_dim = in_channels * self.roi_feat_area
        branch_fcs = nn.ModuleList()
        for i in range(num_branch_fcs):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim

    def forward(self, x):
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)
        return cls_score, bbox_pred


class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super().__init__(
            num_shared_fcs=2, fc_out_channels=fc_out_channels, *args, **kwargs)
