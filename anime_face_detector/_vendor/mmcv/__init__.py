# Minimal replacements for the mmcv building blocks the vendored mmdet
# code depends on. See conv_module.py and nms.py for details.
from .conv_module import ConvModule
from .nms import batched_nms, nms

__all__ = ['ConvModule', 'batched_nms', 'nms']
