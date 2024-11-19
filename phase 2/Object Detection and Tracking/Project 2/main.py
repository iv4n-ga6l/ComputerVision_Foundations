"""
Develop an object tracker using the KLT(Kanade-Lucas-Tomasi) tracker for face tracking in a video stream.
"""

import cv2
import numpy as np

class FaceKLTTracker:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Parameters for KLT tracker
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
        
        # Initialize tracking variables
        self.track_points = None
        self.face_rect = None
        self.prev_frame = None
        
    def detect_face(self, frame):
        """Detect face in the frame and return the largest face rectangle"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            areas = [w * h for (x, y, w, h) in faces]
            max_idx = np.argmax(areas)
            return faces[max_idx]
        return None
    
    def init_tracker(self, frame, face_rect):
        """Initialize tracker with points inside the face region"""
        x, y, w, h = face_rect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask for the face region
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        
        # Detect good features to track inside the face region
        points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        if points is not None:
            self.track_points = points
            self.prev_frame = gray
            self.face_rect = face_rect
            return True
        return False
    
    def update(self, frame):
        """Update tracker with new frame"""
        if self.track_points is None or len(self.track_points) == 0:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, 
            self.track_points, 
            None,
            **self.lk_params
        )
        
        if new_points is None:
            return None
        
        # Keep only good points
        good_new = new_points[status == 1]
        good_old = self.track_points[status == 1]
        
        if len(good_new) < 10:  # If too few points, reinitialize
            return None
        
        # Update face rectangle based on point movement
        if len(good_new) > 0 and len(good_old) > 0:
            movement = good_new - good_old
            mean_movement = np.mean(movement, axis=0)
            
            x, y, w, h = self.face_rect
            x += mean_movement[0]
            y += mean_movement[1]
            
            self.face_rect = (int(x), int(y), w, h)
            
        # Update tracking points and previous frame
        self.track_points = good_new.reshape(-1, 1, 2)
        self.prev_frame = gray
        
        return self.face_rect
    
    def draw_tracking(self, frame):
        """Draw tracking visualization on the frame"""
        if self.face_rect is not None:
            x, y, w, h = self.face_rect
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            
        if self.track_points is not None:
            for point in self.track_points:
                x, y = point.ravel()
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                
        return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    tracker = FaceKLTTracker()
    tracking_initialized = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if not tracking_initialized:
            # Detect face and initialize tracker
            face_rect = tracker.detect_face(frame)
            if face_rect is not None:
                tracking_initialized = tracker.init_tracker(frame, face_rect)
        else:
            # Update tracker
            face_rect = tracker.update(frame)
            if face_rect is None:
                tracking_initialized = False
            
        # Draw tracking visualization
        frame = tracker.draw_tracking(frame)
        
        # Display result
        cv2.imshow('Face Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()