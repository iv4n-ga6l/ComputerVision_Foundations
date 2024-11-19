# Project 4

Develop a Python program that matches logos or objects in a series of images using feature matching.

This program performs feature matching between images using the SIFT (Scale-Invariant Feature Transform) algorithm and OpenCV.
- Uses SIFT (Scale-Invariant Feature Transform) algorithm for robust feature detection
- Implements ratio test for filtering good matches
- Uses RANSAC to find homography matrix between matched points
- Creates visualization of matches and detected objects

To test, prepare your images:
- A template image containing the logo/object you want to find
- A scene image where you want to search for the template

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
python main.py
```