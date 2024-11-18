"""
Build a real-time video frame capture application using OpenCV to record and save video from a webcam.
"""

import cv2
import datetime
from pathlib import Path

class WebcamRecorder:
    def __init__(self, output_dir="recordings"):
        """Initialize the webcam recorder with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 30
        
        self.recording = False
        self.video_writer = None

    def generate_filename(self):
        """Generate a unique filename based on timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"recording_{timestamp}.avi"

    def start_recording(self):
        """Start recording video."""
        if not self.recording:
            filename = self.generate_filename()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                str(filename),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            self.recording = True
            print(f"Recording started: {filename}")

    def stop_recording(self):
        """Stop recording video."""
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("Recording stopped")

    def run(self):
        """Main loop for capturing and recording video."""
        print("Webcam Recorder Started")
        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Recording indicator
                if self.recording:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle
                    cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the frame
                cv2.imshow('Webcam Recorder', frame)

                # Write frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        recorder = WebcamRecorder()
        recorder.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()