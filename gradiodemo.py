import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

import anime_face_detector

torch.hub.download_url_to_file(
    'https://raw.githubusercontent.com/hysts/anime-face-detector/main/assets/input.jpg',
    'input.jpg')

detector = anime_face_detector.create_detector('yolov3')

FACE_SCORE_THRESH = 0.5
LANDMARK_SCORE_THRESH = 0.3


def detect(img):
    image = cv2.imread(img.name)
    preds = detector(image)

    res = image.copy()
    for pred in preds:
        box = pred['bbox']
        box, score = box[:4], box[4]
        if score < FACE_SCORE_THRESH:
            continue
        box = np.round(box).astype(int)

        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

        pred_pts = pred['keypoints']
        for *pt, score in pred_pts:
            if score < LANDMARK_SCORE_THRESH:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(res)
    return image_pil


title = 'hysts/anime-face-detector'
description = 'demo for hysts/anime-face-detector. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.'
article = "<a href='https://github.com/hysts/anime-face-detector'>Github Repo</a>"

gr.Interface(detect, [gr.inputs.Image(type='file', label='Input')],
             gr.outputs.Image(type='pil', label='Output'),
             title=title,
             description=description,
             article=article,
             examples=[
                 ['input.jpg'],
             ],
             enable_queue=True).launch(debug=True, share=True)
