# Copyright (c) OpenMMLab. All rights reserved.
#
# Vendored from mmdetection v2.28.2,
# mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py and
# base_roi_extractor.py, plus bbox2roi from mmdet/core/bbox/transforms.py
# (Apache License 2.0, see ../LICENSE.mmdetection). Modifications:
# removed the registry decorator, fp16/ONNX code and roi_rescale;
# mmcv.ops.RoIAlign is replaced by torchvision.ops.RoIAlign with
# aligned=True and sampling_ratio taken from the config (mmcv's default
# aligned mode, which mmdet 2.x used). The extractor holds no
# parameters, so state_dict keys are unaffected.
import torch
import torch.nn as nn
import torchvision


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        output_size (int): Output size of RoIAlign.
        sampling_ratio (int): Sampling ratio of RoIAlign.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
    """

    def __init__(self,
                 output_size,
                 sampling_ratio,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super().__init__()
        self.roi_layers = nn.ModuleList([
            torchvision.ops.RoIAlign(
                output_size,
                spatial_scale=1 / s,
                sampling_ratio=sampling_ratio,
                aligned=True) for s in featmap_strides
        ])
        self.output_size = output_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        """Forward function."""
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, self.output_size,
            self.output_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
        return roi_feats
