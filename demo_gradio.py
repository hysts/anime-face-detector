import functools
import pathlib

import cv2
import gradio as gr
import numpy as np
import torch

from anime_face_detector import LandmarkDetector, create_detector

REPO_ROOT = pathlib.Path(__file__).parent
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=2)
def _get_detector(detector_name: str) -> LandmarkDetector:
    return create_detector(detector_name, device=DEVICE)


def detect(
    image: np.ndarray,
    detector_name: str,
    face_score_threshold: float,
    landmark_score_threshold: float,
) -> np.ndarray:
    detector = _get_detector(detector_name)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    preds = detector(image_bgr)

    res = image_bgr.copy()
    for pred in preds:
        box = pred["bbox"]
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        line_thickness = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), line_thickness)

        for *pt, pt_score in pred["keypoints"]:
            color = (0, 255, 255) if pt_score < landmark_score_threshold else (0, 0, 255)
            center = np.round(pt).astype(int)
            cv2.circle(res, tuple(center), line_thickness, color, cv2.FILLED)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="anime-face-detector") as demo:
    gr.Markdown("# [anime-face-detector](https://github.com/hysts/anime-face-detector)")
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Input")
            detector_name = gr.Radio(label="Detector", choices=["yolov3", "faster-rcnn"], value="yolov3")
            face_score_threshold = gr.Slider(
                label="Face Score Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5
            )
            landmark_score_threshold = gr.Slider(
                label="Landmark Score Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.3
            )
            run_button = gr.Button("Detect")
        with gr.Column():
            result = gr.Image(label="Output")

    gr.Examples(
        examples=[[(REPO_ROOT / "assets" / "input.jpg").as_posix()]],
        inputs=image,
    )
    run_button.click(
        fn=detect,
        inputs=[image, detector_name, face_score_threshold, landmark_score_threshold],
        outputs=result,
    )

if __name__ == "__main__":
    demo.launch()
