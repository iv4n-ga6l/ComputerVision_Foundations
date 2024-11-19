# Project 2

Develop an object tracker using the KLT(Kanade-Lucas-Tomasi) tracker for face tracking in a video stream.

The FaceKLTTracker class combines:
- Face detection using Haar Cascade Classifier
- Feature point detection using Shi-Tomasi corner detector
- KLT tracking using Lucas-Kanade optical flow

The tracker is robust to:
- Moderate face movements
- Partial occlusions
- Changes in lighting
- Scale changes

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
python main.py
```