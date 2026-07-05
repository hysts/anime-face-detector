# Copyright (c) OpenMMLab. All rights reserved.
#
# Vendored from mmdetection v2.28.2, mmdet/models/necks/fpn.py
# (Apache License 2.0, see ../LICENSE.mmdetection). Modifications:
# removed the registry decorator, fp16 decorator, weight init config and
# the add_extra_convs/end_level options this project's configuration
# does not use (extra levels come from max pooling); ConvModule comes
# from the local mmcv shim. Submodule attribute names are unchanged, so
# state_dict keys match the original.
import torch.nn as nn
import torch.nn.functional as F

from ..mmcv import ConvModule


class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales. Extra output levels beyond
            the backbone levels are produced with stride-2 max pooling.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.backbone_end_level = self.num_ins
        assert num_outs >= self.num_ins - start_level
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=None,
                act_cfg=None,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels with max pooling (e.g., Faster R-CNN)
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
