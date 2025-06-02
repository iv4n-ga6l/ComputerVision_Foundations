# Project 3: PCA Face Recognition (Eigenfaces)

Implement face recognition using Principal Component Analysis (PCA) and the Eigenfaces method.

## Objective
Build a face recognition system using dimensionality reduction techniques to identify faces from a gallery of known individuals.

## Key Concepts

### PCA (Principal Component Analysis):
- Reduces dimensionality while preserving maximum variance
- Projects high-dimensional face images onto lower-dimensional eigenspace
- Identifies the most important features (eigenfaces) in face data
- Computationally efficient for recognition tasks

### Eigenfaces Method:
- Treats face recognition as a 2D pattern recognition problem
- Computes eigenvectors of face covariance matrix
- Each eigenface represents a characteristic face feature
- New faces are projected onto eigenface space for recognition

## Mathematical Foundation
1. **Data Preparation**: Flatten face images into vectors
2. **Mean Subtraction**: Remove average face from all samples
3. **Covariance Matrix**: Compute relationships between pixel intensities
4. **Eigendecomposition**: Find principal components (eigenfaces)
5. **Projection**: Transform faces to eigenface coordinates
6. **Recognition**: Use distance metrics in reduced space

## Features Implemented
- Automatic face detection using Haar cascades
- PCA computation and eigenface visualization
- Face projection to eigenspace
- k-NN classification in reduced dimensions
- Performance evaluation with different numbers of components
- Real-time face recognition from webcam

## Dataset Structure
```
faces_dataset/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── person3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Expected Results
- Visualization of top eigenfaces
- Recognition accuracy: 85-95% (depending on dataset size)
- Real-time face recognition demo
- Analysis of optimal number of principal components

## Applications
- Security and access control systems
- Photo organization and tagging
- Surveillance systems
- Biometric authentication
- Social media face tagging
