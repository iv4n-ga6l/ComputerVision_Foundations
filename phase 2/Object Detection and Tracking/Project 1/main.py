"""
Implement a background subtraction technique to detect moving objects in a video.
"""

import cv2

def detect_motion(video_path, history=500, threshold=16):
    """
    Detect moving objects in a video using background subtraction.
    
    Parameters:
    video_path (str): Path to the video file
    history (int): Number of frames used to build the background model
    threshold (int): Threshold for detecting changes
    """
    # Create a background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=threshold,
        detectShadows=True
    )
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Remove shadows (gray pixels) and noise
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw bounding boxes around moving objects
        for contour in contours:
            # Filter out small contours
            if cv2.contourArea(contour) < 500:  # Adjust this threshold as needed
                continue
                
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw rectangle around moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display results
        cv2.imshow('Original', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        
        # Break loop on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"
    detect_motion(video_path)