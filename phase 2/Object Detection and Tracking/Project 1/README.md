# Project 1

Implement a background subtraction technique to detect moving objects in a video.

This implementation uses OpenCV's MOG2 (Mixture of Gaussians) background subtractor to detect moving objects.

The script creates a MOG2 background subtractor with specified parameters:
- history: Number of frames used to build the background model
- varThreshold: Threshold for detecting changes
- detectShadows: Enable shadow detection


For each frame, it:
- Applies background subtraction to get a foreground mask
- Removes shadows and noise using thresholding
- Applies morphological operations to clean up the mask
- Finds contours of moving objects
- Draws bounding boxes around detected objects


Key features:
- Filters out small contours to reduce false positives
- Shows both the original frame and the foreground mask

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
python main.py
```