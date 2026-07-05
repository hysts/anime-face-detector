# Copyright (c) OpenMMLab. All rights reserved.
#
# Vendored from mmdetection v2.28.2, mmdet/models/dense_heads/rpn_head.py,
# with the test-time driver methods it inherits consolidated from
# mmdet/models/dense_heads/anchor_head.py (__init__),
# mmdet/models/dense_heads/base_dense_head.py (get_bboxes) and
# mmdet/models/dense_heads/dense_test_mixins.py (simple_test_rpn)
# (Apache License 2.0, see ../LICENSE.mmdetection). Modifications:
# removed the registry decorator, losses and all training/ONNX code;
# hardcoded the single-conv head (num_convs=1) and sigmoid
# classification used by this project's configuration; the prior
# generator and bbox coder are constructed directly instead of from
# registry config dicts; test_cfg is a plain dict. Submodule attribute
# names are unchanged, so state_dict keys match the original.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mmcv import batched_nms
from .anchor_generator import AnchorGenerator
from .bbox_coders import DeltaXYWHBBoxCoder


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list


class RPNHead(nn.Module):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
        anchor_generator (AnchorGenerator): The anchor generator.
        bbox_coder (DeltaXYWHBBoxCoder): The box coder.
        test_cfg (dict): Testing config of anchor head, e.g.
            dict(nms_pre=1000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0).
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 anchor_generator,
                 bbox_coder,
                 test_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = 1
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = True
        self.cls_out_channels = self.num_classes

        assert isinstance(anchor_generator, AnchorGenerator)
        assert isinstance(bbox_coder, DeltaXYWHBBoxCoder)
        self.prior_generator = anchor_generator
        self.bbox_coder = bbox_coder
        self.test_cfg = test_cfg

        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=False)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network."""
        cls_scores = []
        bbox_preds = []
        for feat in feats:
            cls_score, bbox_pred = self.forward_single(feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
        return proposal_list

    def get_bboxes(self, cls_scores, bbox_preds, img_metas):
        """Transform network outputs of a batch into bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            img_metas (list[dict]): Image meta info.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5).
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              mlvl_priors, img_meta)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, mlvl_anchors,
                           img_meta):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = copy.deepcopy(self.test_cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            rpn_cls_score = rpn_cls_score.reshape(-1)
            scores = rpn_cls_score.sigmoid()
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg['min_bbox_size'] >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg['min_bbox_size']) & (h > cfg['min_bbox_size'])
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg['nms'])
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg['max_per_img']]
