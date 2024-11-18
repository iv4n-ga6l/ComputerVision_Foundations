# Project 3

Develop a command-line tool that resizes, rotates, and crops images.

## Install
```sh
pip install -r requirements.txt
```

## Run
```sh
# Resize to specific width while maintaining aspect ratio
python main.py horse.jpg output.jpg --width 800

# Resize by scale factor
python main.py horse.jpg output.jpg --scale 0.5

# Rotate image by 90 degrees
python main.py horse.jpg output.jpg --rotate 90

# Crop image
python main.py horse.jpg output.jpg --crop 100 100 500 500

# Combine operations
python main.py horse.jpg output.jpg --width 800 --rotate 45 --crop 100 100 500 500
```