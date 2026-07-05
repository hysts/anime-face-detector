# Third-party licenses

This project is released under the MIT License (see LICENSE), with the
following exception.

## Vendored OpenMMLab code

The code under `anime_face_detector/_vendor/` is derived from OpenMMLab
projects and is licensed under the Apache License 2.0:

- `anime_face_detector/_vendor/mmpose/`: from
  [mmpose](https://github.com/open-mmlab/mmpose) v0.29.0.
  License text: `anime_face_detector/_vendor/LICENSE.mmpose`.
- `anime_face_detector/_vendor/mmdet/`: from
  [mmdetection](https://github.com/open-mmlab/mmdetection) v2.28.2.
  License text: `anime_face_detector/_vendor/LICENSE.mmdetection`.
- `anime_face_detector/_vendor/mmcv/`: minimal reimplementations of
  building blocks from [mmcv](https://github.com/open-mmlab/mmcv) v1.7.0
  (Apache License 2.0).

Each vendored file keeps its upstream copyright header and states the
exact source file and the modifications made. The vendored versions are
the ones the pretrained checkpoints were trained with.
