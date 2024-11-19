# Project 5

Implement an application to detect and track vehicles on a highway video using Mean Shift or CamShift.

This implementation combines background subtraction for initial vehicle detection with CamShift for tracking. Here's how it works:
- Uses MOG2 background subtractor to detect moving vehicles
- Implements CamShift tracking for following vehicles across frames

Key features:
- Background subtraction with noise removal
- Contour detection and filtering based on size
- HSV color space tracking for better robustness

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
python main.py
```