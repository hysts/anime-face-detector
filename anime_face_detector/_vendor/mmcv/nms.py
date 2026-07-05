# Copyright (c) OpenMMLab. All rights reserved.
#
# Minimal reimplementation of mmcv v1.7.0's mmcv/ops/nms.py nms() and
# batched_nms() (Apache License 2.0; mmcv uses the same license text as
# ../LICENSE.mmpose), delegating the core suppression to
# torchvision.ops.nms instead of mmcv's compiled extension. Only the
# behavior exercised by the vendored mmdet test paths is kept: plain
# 'nms' type, axis-aligned (N, 4) boxes, no split_thr chunking.
import torch
import torchvision


def nms(boxes, scores, iou_threshold):
    """NMS returning (dets, keep) like mmcv.ops.nms.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        tuple: kept dets (boxes and scores, shape (K, 5)) and indice,
        sorted by score in descending order.
    """
    inds = torchvision.ops.nms(boxes, scores, iou_threshold)
    dets = torch.cat([boxes[inds], scores[inds].reshape(-1, 1)], dim=1)
    return dets, inds


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    In order to perform NMS independently per class, an offset is added
    to all the boxes. The offset is dependent only on the class idx, and
    is large enough so that boxes from different classes do not overlap.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | None): When None, NMS is skipped and the boxes are
            just sorted by score. Otherwise a dict like
            dict(type='nms', iou_threshold=0.5); only type 'nms' is
            supported.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets (boxes and scores, shape (K, 5)) and indice.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    nms_type = nms_cfg_.pop('type', 'nms')
    assert nms_type == 'nms', f'unsupported nms type {nms_type}'

    if class_agnostic:
        boxes_for_nms = boxes
    elif boxes.numel() > 0:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
    else:
        boxes_for_nms = boxes

    dets, keep = nms(boxes_for_nms, scores, **nms_cfg_)
    boxes = boxes[keep]
    scores = dets[:, -1]
    return torch.cat([boxes, scores[:, None]], -1), keep
