"""Golden-output generator for the pure PyTorch migration.

This script runs ONLY in the one-time legacy reference environment
(Python 3.9, torch 1.13.1+cpu, mmcv-full 1.7.0, mmdet 2.28.2,
mmpose 0.29.0) against the original mm*-based implementation at commit
7bcc2ae. It captures reference outputs and intermediates as .npz files
that the new implementation is tested against. See tests/golden/README.md.

Usage (from the repo root, inside the legacy env):

    PYTHONPATH=. python tests/golden/generate_golden.py
"""

import pathlib

import cv2
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
IMAGE_DIR = GOLDEN_DIR / "images"

DETECTOR_NAMES = ["faster-rcnn", "yolov3"]
# Crop window (x0, y0, x1, y1) in input.jpg holding a single face,
# picked so the face found by both detectors sits well inside.
CROP_WINDOW = (2100, 1450, 2600, 1950)
SMALL_SCALE = 0.2
MAX_LANDMARK_BOXES = 5


def prepare_images() -> dict:
    """Create the derived test images (committed alongside the goldens)."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    full = cv2.imread((REPO_ROOT / "assets" / "input.jpg").as_posix())
    assert full is not None

    small_path = IMAGE_DIR / "small.png"
    if not small_path.exists():
        small = cv2.resize(full, None, fx=SMALL_SCALE, fy=SMALL_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(small_path.as_posix(), small)

    crop_path = IMAGE_DIR / "crop.png"
    if not crop_path.exists():
        x0, y0, x1, y1 = CROP_WINDOW
        cv2.imwrite(crop_path.as_posix(), full[y0:y1, x0:x1])

    noface_path = IMAGE_DIR / "noface.png"
    if not noface_path.exists():
        # Deterministic gradient image with no face in it.
        yy, xx = np.mgrid[0:480, 0:640]
        noface = np.stack(
            [
                (yy * 255 // 479).astype(np.uint8),
                (xx * 255 // 639).astype(np.uint8),
                ((yy + xx) * 255 // (479 + 639)).astype(np.uint8),
            ],
            axis=-1,
        )
        cv2.imwrite(noface_path.as_posix(), noface)

    return {
        "input": cv2.imread((REPO_ROOT / "assets" / "input.jpg").as_posix()),
        "small": cv2.imread(small_path.as_posix()),
        "crop": cv2.imread(crop_path.as_posix()),
        "noface": cv2.imread(noface_path.as_posix()),
    }


def set_flip_test(detector, flag: bool) -> None:
    """Toggle flip_test the same way the original detector.py does."""
    detector.landmark_detector.cfg.model.test_cfg.flip_test = flag
    # The built module holds the same (shared) cfg object; make sure the
    # mutation actually reached it, otherwise the golden would be wrong.
    assert detector.landmark_detector.test_cfg.get("flip_test") is flag


def tensor_summary(prefix: str, tensor: torch.Tensor, out: dict) -> None:
    array = tensor.detach().cpu().numpy()
    out[f"{prefix}/shape"] = np.array(array.shape, dtype=np.int64)
    out[f"{prefix}/mean"] = np.float64(array.mean())
    out[f"{prefix}/std"] = np.float64(array.std())
    out[f"{prefix}/crop"] = array[0, :2, :6, :6].astype(np.float32)


def summarize_hook(store: dict, prefix: str):
    def hook(_module, _inputs, outputs):
        tensors = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        # YOLOV3Head returns a 1-element tuple of a list of pred maps.
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]
        for i, t in enumerate(tensors):
            if isinstance(t, torch.Tensor):
                tensor_summary(f"{prefix}/{i}", t, store)

    return hook


def generate_detector_goldens(images: dict) -> dict:
    """Raw and scaled boxes per detector per image, plus e2e keypoints."""
    from mmdet.apis import inference_detector

    from anime_face_detector import create_detector

    fixed_boxes = {}
    for det_name in DETECTOR_NAMES:
        detector = create_detector(det_name, device="cpu")
        det_data = {}
        e2e_data = {}
        for image_name, image in images.items():
            raw_boxes = inference_detector(detector.face_detector, image)[0]
            det_data[f"{image_name}/raw_boxes"] = raw_boxes.astype(np.float32)
            scaled = detector._update_pred_box(raw_boxes.copy())
            scaled = np.stack(scaled) if scaled else np.zeros((0, 5), np.float32)
            det_data[f"{image_name}/scaled_boxes"] = scaled.astype(np.float32)

            for flip in [True, False]:
                set_flip_test(detector, flip)
                preds = detector(image)
                key = "flip" if flip else "noflip"
                bboxes = np.stack([p["bbox"] for p in preds]) if preds else np.zeros((0, 5), np.float32)
                kpts = np.stack([p["keypoints"] for p in preds]) if preds else np.zeros((0, 28, 3), np.float32)
                e2e_data[f"{image_name}/bboxes_{key}"] = bboxes.astype(np.float32)
                e2e_data[f"{image_name}/keypoints_{key}"] = kpts.astype(np.float32)

            if det_name == "yolov3" and image_name in ["input", "crop"]:
                fixed_boxes[image_name] = scaled[:MAX_LANDMARK_BOXES].astype(np.float32)

        np.savez_compressed(GOLDEN_DIR / f"detector_{det_name}.npz", **det_data)
        np.savez_compressed(GOLDEN_DIR / f"e2e_{det_name}.npz", **e2e_data)
        print(f"wrote detector_{det_name}.npz / e2e_{det_name}.npz")
    return fixed_boxes


def generate_landmark_goldens(images: dict, fixed_boxes: dict) -> None:
    """Landmark goldens decoupled from the detectors: fixed input boxes."""
    from mmpose.apis import inference_top_down_pose_model

    from anime_face_detector import create_detector

    detector = create_detector("yolov3", device="cpu")
    data = {}
    for image_name, boxes in fixed_boxes.items():
        image = images[image_name]
        data[f"{image_name}/boxes"] = boxes
        for flip in [True, False]:
            set_flip_test(detector, flip)
            key = "flip" if flip else "noflip"
            preds, outputs = inference_top_down_pose_model(
                detector.landmark_detector,
                image,
                [{"bbox": box} for box in boxes],
                format="xyxy",
                dataset_info=detector.dataset_info,
                return_heatmap=True,
                outputs=None,
            )
            kpts = np.stack([p["keypoints"] for p in preds])
            data[f"{image_name}/keypoints_{key}"] = kpts.astype(np.float32)
            if image_name == "crop":
                heatmap = outputs[0]["heatmap"]
                data[f"{image_name}/heatmap_{key}"] = np.asarray(heatmap, dtype=np.float32)
    np.savez_compressed(GOLDEN_DIR / "landmarks.npz", **data)
    print("wrote landmarks.npz")


def preprocess_mmdet_image(model, image: np.ndarray):
    """Replicate inference_detector's preprocessing (CPU path)."""
    from mmcv.parallel import collate
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose

    cfg = model.cfg.copy()
    cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    pipeline = Compose(cfg.data.test.pipeline)
    data = pipeline(dict(img=image))
    data = collate([data], samples_per_gpu=1)
    # Same unwrapping as inference_detector's CPU path: MultiScaleFlipAug
    # yields one DataContainer per scale (single scale here).
    img_metas = data["img_metas"][0].data[0]
    img = data["img"][0].data[0]
    return img, img_metas


def generate_frcnn_intermediates(image: np.ndarray) -> None:
    from anime_face_detector import create_detector

    detector = create_detector("faster-rcnn", device="cpu")
    model = detector.face_detector
    store = {}
    hooks = [
        model.backbone.register_forward_hook(summarize_hook(store, "backbone")),
        model.neck.register_forward_hook(summarize_hook(store, "neck")),
    ]
    img, img_metas = preprocess_mmdet_image(model, image)
    store["input/shape"] = np.array(img.shape, dtype=np.int64)
    store["input/mean"] = np.float64(img.numpy().mean())
    store["input/std"] = np.float64(img.numpy().std())
    with torch.no_grad():
        feats = model.extract_feat(img)
        rpn_cls, rpn_reg = model.rpn_head(feats)
        for i, t in enumerate(rpn_cls):
            tensor_summary(f"rpn_cls/{i}", t, store)
        for i, t in enumerate(rpn_reg):
            tensor_summary(f"rpn_reg/{i}", t, store)
        proposals = model.rpn_head.simple_test_rpn(feats, img_metas)
        store["proposals"] = proposals[0][:100].numpy().astype(np.float32)
        det_bboxes, det_labels = model.roi_head.simple_test_bboxes(
            feats, img_metas, proposals, model.roi_head.test_cfg, rescale=True
        )
        store["det_bboxes"] = det_bboxes[0].numpy().astype(np.float32)
        store["det_labels"] = det_labels[0].numpy().astype(np.int64)
    for h in hooks:
        h.remove()
    np.savez_compressed(GOLDEN_DIR / "intermediates_faster-rcnn.npz", **store)
    print("wrote intermediates_faster-rcnn.npz")


def generate_yolov3_intermediates(image: np.ndarray) -> None:
    from anime_face_detector import create_detector

    detector = create_detector("yolov3", device="cpu")
    model = detector.face_detector
    store = {}
    hooks = [
        model.backbone.register_forward_hook(summarize_hook(store, "backbone")),
        model.neck.register_forward_hook(summarize_hook(store, "neck")),
        model.bbox_head.register_forward_hook(summarize_hook(store, "pred_maps")),
    ]
    img, img_metas = preprocess_mmdet_image(model, image)
    store["input/shape"] = np.array(img.shape, dtype=np.int64)
    store["input/mean"] = np.float64(img.numpy().mean())
    store["input/std"] = np.float64(img.numpy().std())
    with torch.no_grad():
        feats = model.extract_feat(img)
        pred_maps = model.bbox_head(feats)[0]
        results = model.bbox_head.get_bboxes(pred_maps, img_metas, rescale=True)
        store["det_bboxes"] = results[0][0].numpy().astype(np.float32)
    for h in hooks:
        h.remove()
    np.savez_compressed(GOLDEN_DIR / "intermediates_yolov3.npz", **store)
    print("wrote intermediates_yolov3.npz")


def generate_hrnetv2_intermediates(image: np.ndarray, fixed_boxes: dict) -> None:
    from mmpose.apis import inference_top_down_pose_model

    from anime_face_detector import create_detector

    detector = create_detector("yolov3", device="cpu")
    set_flip_test(detector, False)
    model = detector.landmark_detector
    store = {}
    hooks = [
        model.backbone.register_forward_hook(summarize_hook(store, "backbone")),
    ]
    inference_top_down_pose_model(
        model, image, [{"bbox": fixed_boxes["crop"][0]}], format="xyxy", dataset_info=detector.dataset_info
    )
    for h in hooks:
        h.remove()
    np.savez_compressed(GOLDEN_DIR / "intermediates_hrnetv2.npz", **store)
    print("wrote intermediates_hrnetv2.npz")


def main() -> None:
    import argparse

    import mmcv
    import mmdet
    import mmpose

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", choices=["main", "intermediates"], default=None, help="Regenerate only a subset of the goldens."
    )
    args = parser.parse_args()

    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    images = prepare_images()
    if args.only != "intermediates":
        fixed_boxes = generate_detector_goldens(images)
        generate_landmark_goldens(images, fixed_boxes)
    else:
        with np.load(GOLDEN_DIR / "landmarks.npz") as data:
            fixed_boxes = {"crop": data["crop/boxes"]}
    if args.only != "main":
        generate_frcnn_intermediates(images["crop"])
        generate_yolov3_intermediates(images["crop"])
        generate_hrnetv2_intermediates(images["crop"], fixed_boxes)

    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "mmcv": mmcv.__version__,
        "mmdet": mmdet.__version__,
        "mmpose": mmpose.__version__,
    }
    np.savez_compressed(GOLDEN_DIR / "environment.npz", **{k: np.str_(v) for k, v in versions.items()})
    print("wrote environment.npz:", versions)


if __name__ == "__main__":
    main()
