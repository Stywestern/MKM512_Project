# modules/interface.py

##################################### Imports #####################################
# Libraries
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time

# Modules
import config
from modules.utils import log
from modules.detector import YOLODetector
#from modules.recognizer import TurretRecognizer

###################################################################################

class VisionWorker(QThread):
    # Signals to communicate with the UI
    # Sends: [Main Frame, Aligned Face, Data Dict]
    update_signal = pyqtSignal(np.ndarray, np.ndarray, dict)

    def __init__(self, camera_instance):
        super().__init__()
        self.cam = camera_instance # Use the pre-started camera
        self.detector = YOLODetector()
        #self.recognizer = TurretRecognizer()
        self.running = True
        self.is_frozen = False

        log("VisionWorker initialized", "INFO")

    def run(self):
        prev_time = 0
        log("Running Sentry Logic Subsystem", "INFO")
        
        while self.running:
            start_time = time.time() # Start the clock for this loop

            frame = self.cam.read() # latest frame
            if frame is None or frame.size == 0:
                self.msleep(10) 
                continue

            # 2. Flow Control: If frozen, we don't process AI or emit new signals
            if not self.is_frozen:
                # Placeholders for the Aligned Face and Cosine Scores
                history_msg = ""

                # --- The Vision Pipeline ---
                # 1. YOLO Detection
                detections = self.detector.detect_and_track(frame)
                if detections:
                    # 1. Get the raw results
                    target = detections[0]
                    x1, y1, x2, y2 = target["face_bbox"]
                    track_id = target["id"]

                    # 2. We prevent the app from crashing if YOLO predicts outside the frame
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # 3. THE CROP
                    detect_crop = frame[y1:y2, x1:x2].copy()

                # 2. Tracking
                # 3. Recognition (Adaptive Padding logic we built)

                # Package the telemetry
                data = {
                    "fps": round(fps, 1),
                    "status": "TRACKING" if not self.is_frozen else "FROZEN",
                    "counts": 0 # Placeholder for detected faces
                }

                # Push UI
                    # Fps
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
                prev_time = current_time
                    # Send data
                self.update_signal.emit(frame, detect_crop, data)

            elapsed = time.time() - start_time
            sleep_time = max(1, int((0.033 - elapsed) * 1000)) 
            self.msleep(sleep_time)

    def step_forward(self):
        print("I stepped forward")

    def step_backward(self):
        print("I stepped backward")

    def reset_tracking_data(self):
        print("I reset")

    def stop(self):
        self.running = False
        self.cam.stop()

        log("Sentry Logic Subsystem Stopped", "INFO")

        self.wait()
