# gfx803_nms_fix.py
# torchvision's compiled nms/batched_nms extension has no gfx803 code object
# on this build ("HIP error: invalid device function" -- a missing-kernel
# class of failure, confirmed via a standalone torchvision.ops.nms repro,
# independent of ultralytics/ComfyUI). Route it through the CPU
# implementation when called on a CUDA/HIP tensor -- NMS runs on a small
# number of boxes, so the host round-trip is negligible.
# WORKAROUND, not a fix: the source fix is building torchvision's ops with
# gfx803 code objects in the Dockerfile's pytorch-builder stage.
import torch
import torchvision
from torchvision.ops import boxes as _tv_boxes_module

_orig_nms = torchvision.ops.nms
_orig_batched_nms = torchvision.ops.batched_nms

def _safe_nms(boxes, scores, iou_threshold):
    if not boxes.is_cuda:
        return _orig_nms(boxes, scores, iou_threshold)
    device = boxes.device
    keep = _orig_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
    return keep.to(device)

def _safe_batched_nms(boxes, scores, idxs, iou_threshold):
    if not boxes.is_cuda:
        return _orig_batched_nms(boxes, scores, idxs, iou_threshold)
    device = boxes.device
    keep = _orig_batched_nms(boxes.detach().cpu(), scores.detach().cpu(), idxs.detach().cpu(), iou_threshold)
    return keep.to(device)

torchvision.ops.nms = _safe_nms
torchvision.ops.batched_nms = _safe_batched_nms
_tv_boxes_module.nms = _safe_nms
_tv_boxes_module.batched_nms = _safe_batched_nms
