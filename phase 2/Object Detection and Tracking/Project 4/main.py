"""
Create a Python tool that uses optical flow to visualize the movement of objects in a video.
"""

import cv2
import numpy as np

class OpticalFlowVisualizer:
    def __init__(self):
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Random colors for flow visualization
        self.color = np.random.randint(0, 255, (100, 3))
        
    def process_video(self, input_path, output_path=None):
        """
        Process video file and visualize optical flow
        
        Args:
            input_path (str): Path to input video file
            output_path (str, optional): Path to save output video. If None, display only.
        """
        cap = cv2.VideoCapture(input_path)
        
        # Read first frame
        ret, old_frame = cap.read()
        if not ret:
            print("Failed to read video")
            return
            
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        
        # Get video properties for output
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, 
                frame_gray, 
                p0, 
                None, 
                **self.lk_params
            )
            
            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
            
            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Draw line between old and new position
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                               self.color[i].tolist(), 2)
                
                # Draw filled circle at new position
                frame = cv2.circle(frame, (int(a), int(b)), 5, 
                                 self.color[i].tolist(), -1)
            
            # Combine frame with flow visualization
            img = cv2.add(frame, mask)
            
            # Display or write frame
            if output_path:
                out.write(img)
            else:
                cv2.imshow('Optical Flow', img)
                
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Update previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    flow_visualizer = OpticalFlowVisualizer()
    flow_visualizer.process_video('video.mp4', 'output_video.mp4')