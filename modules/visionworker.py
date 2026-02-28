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
from modules.recognizer import TurretRecognizer

###################################################################################

class VisionWorker(QThread):
    # Signals to communicate with the UI
    # Sends: [Main Frame, Yolo Crop, Data Dict]
    update_signal = pyqtSignal(np.ndarray, np.ndarray, dict)

    def __init__(self, camera_instance):
        super().__init__()
        self.cam = camera_instance # Use the pre-started camera
        self.detector = YOLODetector()
        self.recognizer = TurretRecognizer()

        self.active_targets = {} 

        self.running = True
        self.is_frozen = False

        log("VisionWorker initialized", "INFO")

    def run(self):
        prev_time = 0
        log("Running Sentry Logic Subsystem", "INFO")
        
        while self.running:
            start_time = time.time()
            new_snap = np.array([], dtype=np.uint8) # Initialize as such as it expects an array
            
            # 1. Capture
            frame = self.cam.read()
            if frame is None or frame.size == 0:
                self.msleep(10)
                continue

            # 2. Scan
            if not self.is_frozen:
                # Step A: YOLO Detection & Tracking
                detections = self.detector.detect_and_track(frame)
                
                for target in detections:
                    track_id = target["id"]
                    x1, y1, x2, y2 = target["face_bbox"]

                    # POSSIBILITY 1: Brand New Target
                    if track_id not in self.active_targets:
                        # 1. Clean Crop (Before drawing the box)
                        h, w = frame.shape[:2]
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(w, x2), min(h, y2)
                        new_snap = frame[y1c:y2c, x1c:x2c].copy()

                        # 2. Trigger Recognition/Alignment Pipeline
                        # ... RetinaFace / ArcFace logic ...
                        
                        # 3. Latch into Memory
                        self.active_targets[track_id] = "Unknown"
                        update_log = f"New Target: ID {track_id}, {self.active_targets[track_id]}"
                    
                    # POSSIBILITY 2: Known Target
                    else:
                        pass

                    name = self.active_targets.get(track_id, "Unknown")

                    # 2. DRAWING: Box + Name Tag
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # box
                    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1) # header
                    cv2.putText(frame, f"{name} (ID:{track_id})", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 3. TELEMETRY & EMIT
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            data = {
                "fps": round(fps, 1),
                "update": update_log if update_log is not None else None
            }

            # Emit the frame (with all boxes) and the single new snapshot (if any)
            if not self.is_frozen:
                self.update_signal.emit(frame, new_snap, data)

            # Maintaining the 30FPS ceiling so we don't overwhelm the UI thread
            elapsed = time.time() - start_time
            sleep_time = max(1, int((0.033 - elapsed) * 1000)) 
            self.msleep(sleep_time)


    def step_forward(self):
        print("I stepped forward")

    def step_backward(self):
        print("I stepped backward")

    def reset_tracking_data(self):
        print("I reset")
