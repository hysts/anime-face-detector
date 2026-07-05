# Copyright (c) OpenMMLab. All rights reserved.
#
# Minimal reimplementation of mmcv v1.7.0's
# mmcv/cnn/bricks/conv_module.py (Apache License 2.0, see
# ../LICENSE.mmpose for the license text; mmcv uses the same license).
# Only the features used by the vendored mmdet modules are kept:
# Conv2d + optional BatchNorm2d + optional ReLU/LeakyReLU in
# ('conv', 'norm', 'act') order, with bias='auto' semantics. The
# submodule attribute names ('conv', 'bn', 'activate') match mmcv, so
# state_dict keys are identical.
import torch.nn as nn


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Default: 1.
        bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias will be set as True if `norm_cfg` is None,
            otherwise False. Default: "auto".
        norm_cfg (dict): Config dict for normalization layer. Only
            dict(type='BN') is supported. Default: None.
        act_cfg (dict): Config dict for activation layer. Only
            dict(type='ReLU') and dict(type='LeakyReLU') are supported.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias='auto',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True):
        super().__init__()
        assert norm_cfg is None or norm_cfg['type'] == 'BN'
        assert act_cfg is None or act_cfg['type'] in ['ReLU', 'LeakyReLU']

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if self.with_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        if self.with_activation:
            if act_cfg['type'] == 'ReLU':
                self.activate = nn.ReLU(inplace=inplace)
            else:
                self.activate = nn.LeakyReLU(
                    negative_slope=act_cfg.get('negative_slope', 0.01),
                    inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_activation:
            x = self.activate(x)
        return x
