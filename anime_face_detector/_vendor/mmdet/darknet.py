# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
#
# Vendored from mmdetection v2.28.2, mmdet/models/backbones/darknet.py
# (Apache License 2.0, see ../LICENSE.mmdetection). Modifications:
# removed the registry decorator, weight init and training-only code
# (stage freezing, norm_eval handling); ConvModule comes from the local
# mmcv shim with the BN + LeakyReLU(0.1) configuration hardcoded.
# Submodule attribute names are unchanged, so state_dict keys match the
# original.

import torch.nn as nn

from ..mmcv import ConvModule

NORM_CFG = dict(type='BN')
ACT_CFG = dict(type='LeakyReLU', negative_slope=0.1)


class ResBlock(nn.Module):
    """The basic residual block used in Darknet. Each ResBlock consists of two
    ConvModules and the input is added to the final output. Each ConvModule is
    composed of Conv, BN, and LeakyReLU. In YoloV3 paper, the first convLayer
    has half of the number of the filters as much as the second convLayer. The
    first convLayer has filter size of 1x1 and the second one has the filter
    size of 3x3.

    Args:
        in_channels (int): The input channels. Must be even.
    """

    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2

        cfg = dict(norm_cfg=NORM_CFG, act_cfg=ACT_CFG)

        self.conv1 = ConvModule(in_channels, half_in_channels, 1, **cfg)
        self.conv2 = ConvModule(
            half_in_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out


class Darknet(nn.Module):
    """Darknet backbone.

    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                               (512, 1024)))
    }

    def __init__(self, depth=53, out_indices=(3, 4, 5)):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')

        self.depth = depth
        self.out_indices = out_indices
        self.layers, self.channels = self.arch_settings[depth]

        self.conv1 = ConvModule(
            3, 32, 3, padding=1, norm_cfg=NORM_CFG, act_cfg=ACT_CFG)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(layer_name,
                            self.make_conv_res_block(in_c, out_c, n_layers))
            self.cr_blocks.append(layer_name)

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    @staticmethod
    def make_conv_res_block(in_channels, out_channels, res_repeat):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
        """
        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=NORM_CFG,
                act_cfg=ACT_CFG))
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx), ResBlock(out_channels))
        return model
