# Project 1

Implement an edge detection algorithm using the Sobel operator and compare it with the Canny edge detection method.

Key differences between Sobel and Canny edge detection:

### Sobel Operator:
- Computes image gradients using derivative filters
- Detects edges based on intensity changes
- Less sophisticated noise handling
- Provides gradient magnitude information

### Canny Edge Detection:
- More advanced multi-stage algorithm
- Includes noise reduction using Gaussian filtering
- Uses hysteresis thresholding to detect strong and weak edges
- Suppresses non-maximum edges
- Generally produces cleaner, more precise edge maps

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
python main.py
```