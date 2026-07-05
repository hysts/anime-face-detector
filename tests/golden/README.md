# Golden reference outputs

These files capture the behavior of the original mm*-based implementation (commit 7bcc2ae) and gate the pure PyTorch reimplementation. They were generated once, on CPU, in a legacy environment that is not part of this repo's setup:

- Python 3.9, torch 1.13.1+cpu, torchvision 0.14.1+cpu, numpy 1.23.5
- mmcv-full 1.7.0, mmdet 2.28.2, mmpose 0.29.0

See `environment.npz` for the recorded versions and `generate_golden.py` for the exact procedure. The parity tests in `tests/` load these files; they never need the legacy environment.

## Test images

- `../../assets/input.jpg`: 3072x3072, dozens of faces (`input`)
- `images/small.png`: the same image downscaled to 0.2x, small faces (`small`)
- `images/crop.png`: a 500x500 single-face crop of input.jpg (`crop`)
- `images/noface.png`: deterministic gradient, no faces (`noface`)

## Files

- `detector_{faster-rcnn,yolov3}.npz`: per image, `{name}/raw_boxes` as returned by the mmdet detector (score threshold 0.05) and `{name}/scaled_boxes` after `_update_pred_box` with box_scale_factor 1.1.
- `e2e_{faster-rcnn,yolov3}.npz`: full-pipeline results per image, `{name}/bboxes_{flip,noflip}` and `{name}/keypoints_{flip,noflip}` for flip_test on/off.
- `landmarks.npz`: landmark goldens decoupled from the detectors. `{name}/boxes` are the fixed input boxes fed to the pose model, with `{name}/keypoints_{flip,noflip}` outputs, plus raw output heatmaps (post flip-averaging, pre decoding) for the crop image.
- `intermediates_{faster-rcnn,yolov3,hrnetv2}.npz`: layer-by-layer debugging aids captured on the crop image: per-level tensor summaries (shape, mean, std, a `[0, :2, :6, :6]` crop) of backbone/neck/head outputs, exact RPN proposals (first 100), and final detections.

## Tolerances

The new implementation must reproduce, per the migration plan: landmark coordinates within 0.5 px max abs diff and scores within 1e-3; detector boxes within 1 px with the same detection count at score threshold 0.5.
