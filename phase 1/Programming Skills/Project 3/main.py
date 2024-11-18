"""
Develop a command-line tool that resizes, rotates, and crops images.
"""

import cv2
import argparse
import os
from typing import Optional

class ImageProcessor:
    def __init__(self, image_path: str):
        """Initialize the image processor with an image path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.original_image = self.image.copy()
    
    def resize(self, width: Optional[int] = None, height: Optional[int] = None, scale: Optional[float] = None) -> None:
        """
        Resize the image based on width, height, or scale factor.
        
        Args:
            width: Target width in pixels
            height: Target height in pixels
            scale: Scale factor (e.g., 0.5 for half size)
        """
        if scale is not None:
            self.image = cv2.resize(self.image, None, fx=scale, fy=scale)
        elif width is not None and height is not None:
            self.image = cv2.resize(self.image, (width, height))
        elif width is not None:
            aspect_ratio = width / self.image.shape[1]
            height = int(self.image.shape[0] * aspect_ratio)
            self.image = cv2.resize(self.image, (width, height))
        elif height is not None:
            aspect_ratio = height / self.image.shape[0]
            width = int(self.image.shape[1] * aspect_ratio)
            self.image = cv2.resize(self.image, (width, height))
    
    def rotate(self, angle: float, maintain_size: bool = True) -> None:
        """
        Rotate the image by the specified angle.
        
        Args:
            angle: Rotation angle in degrees
            maintain_size: If True, maintain original image size
        """
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        if maintain_size:
            self.image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
        else:
            # Calculate new dimensions
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            self.image = cv2.warpAffine(self.image, rotation_matrix, (new_width, new_height))
    
    def crop(self, x: int, y: int, width: int, height: int) -> None:
        """
        Crop the image to the specified rectangle.
        
        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            width: Width of crop region
            height: Height of crop region
        """
        self.image = self.image[y:y+height, x:x+width]
    
    def save(self, output_path: str) -> None:
        """Save the processed image."""
        cv2.imwrite(output_path, self.image)

def main():
    parser = argparse.ArgumentParser(description='Process images with resize, rotate, and crop operations')
    parser.add_argument('input_image', help='Path to input image')
    parser.add_argument('output_image', help='Path to save output image')
    
    # Resize arguments
    resize_group = parser.add_argument_group('resize options')
    resize_group.add_argument('--width', type=int, help='Target width for resize')
    resize_group.add_argument('--height', type=int, help='Target height for resize')
    resize_group.add_argument('--scale', type=float, help='Scale factor for resize')
    
    # Rotate arguments
    rotate_group = parser.add_argument_group('rotate options')
    rotate_group.add_argument('--rotate', type=float, help='Rotation angle in degrees')
    rotate_group.add_argument('--maintain-size', action='store_true', help='Maintain original size after rotation')
    
    # Crop arguments
    crop_group = parser.add_argument_group('crop options')
    crop_group.add_argument('--crop', nargs=4, type=int, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                           help='Crop rectangle coordinates: x y width height')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = ImageProcessor(args.input_image)
        
        # Apply operations in sequence: resize -> rotate -> crop
        if any([args.width, args.height, args.scale]):
            processor.resize(args.width, args.height, args.scale)
        
        if args.rotate is not None:
            processor.rotate(args.rotate, args.maintain_size)
        
        if args.crop is not None:
            processor.crop(*args.crop)
        
        # Save result
        processor.save(args.output_image)
        print(f"Image processed successfully and saved to {args.output_image}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()