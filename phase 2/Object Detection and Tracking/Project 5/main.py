"""
Implement an application to detect and track vehicles on a highway video using Mean Shift or CamShift.
"""

import cv2
import numpy as np

class VehicleTracker:
    def __init__(self):
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        
        # Initialize parameters for CamShift
        self.track_window = None
        self.roi_hist = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # Tracking state
        self.tracking = False
        self.vehicles = []
        
    def detect_vehicles(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        min_area = 500
        vehicle_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return vehicle_contours
    
    def initialize_tracking(self, frame, contour):
        x, y, w, h = cv2.boundingRect(contour)
        self.track_window = (x, y, w, h)
        
        # Set up the ROI for tracking
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask and calculate histogram
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        self.tracking = True
    
    def track_vehicle(self, frame):
        if not self.tracking:
            return None
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # Apply CamShift
        ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)
        
        # Draw tracking result
        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)
        return pts

def main():
    cap = cv2.VideoCapture("video4.mp4")  # Replace 0 with video file path
    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output4.mp4', fourcc, fps, (frame_width, frame_height))

    tracker = VehicleTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect vehicles
        vehicle_contours = tracker.detect_vehicles(frame)
        
        # Draw contours
        cv2.drawContours(frame, vehicle_contours, -1, (0, 255, 0), 2)
        
        # Initialize tracking for new vehicles
        for contour in vehicle_contours:
            if not tracker.tracking:
                tracker.initialize_tracking(frame, contour)
        
        # Track vehicles
        # if tracker.tracking:
        #     tracked_pts = tracker.track_vehicle(frame)
        #     if tracked_pts is not None:
        #         cv2.polylines(frame, [tracked_pts], True, (0, 0, 255), 2)
        
        out.write(frame)
        
        # Display result
        cv2.imshow('Vehicle Tracking', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()