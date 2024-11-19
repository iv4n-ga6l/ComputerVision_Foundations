# Project 3

Build a basic object detection system using template matching to find logos in a series of images.

The system includes the following features:
- Template matching using OpenCV's matchTemplate function
- Non-maximum suppression to remove overlapping detections
- Batch processing of multiple images
- Visualization of results
- Configurable matching threshold
- Error handling for image loading and processing

Some limitations to be aware of:
- Template matching is sensitive to scale and rotation
- Works best with logos that maintain consistent appearance
- May have false positives with similar patterns
- Performance depends on image and template sizes

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
python main.py
```