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
    update_signal = pyqtSignal(np.ndarray, list, list)

    def __init__(self, camera_instance):
        super().__init__()
        self.cam = camera_instance # Use the pre-started camera
        self.detector = YOLODetector()
        self.recognizer = TurretRecognizer()

        self.active_targets = {} 

        self.running = True
        self.is_frozen = False

        self.locked_target_id = None  # ID of the current "Enemy"
        self.is_firing = False

        log("VisionWorker initialized", "INFO")

    def run(self):
        prev_time = 0
        log("Running Sentry Logic Subsystem", "INFO")
        
        while self.running:
            loop_start = time.time()
            fps = 1 / (loop_start - prev_time + 0.00000000000000001)

            prev_time = loop_start
            
            # --- POSSIBILITY 1: No Face Detected (Just send the frame) ---
            empty_img = np.array([], dtype=np.uint8)
            image_package = [empty_img, empty_img] # [YOLO_CROP, ALIGN_CROP]
            update_log = None
            distances = {} 
            
            # 1. Capture
            frame = self.cam.read()
            if frame is None or frame.size == 0:
                self.msleep(10); continue

            # 2. Scan
            if not self.is_frozen:
                detections = self.detector.detect_and_track(frame)

                if self.locked_target_id is None and len(detections) > 0:
                    self.locked_target_id = detections[0]["id"]
                    log(f"TARGET ACQUIRED: ID {self.locked_target_id}", "WARNING")
                
                for target in detections:
                    track_id = target["id"]

                    is_enemy = (track_id == self.locked_target_id)
                    color = (0, 0, 255) if is_enemy else (0, 255, 0)
                    thickness = 4 if (is_enemy and self.is_firing) else 2

                    x1, y1, x2, y2 = target["face_bbox"] # yolo box is here

                    # --- POSSIBILITY 3: Brand New Target (Send frame, [yolo, aligned]) ---
                    if track_id not in self.active_targets:
                        h, w = frame.shape[:2]
                        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2) # border protection
                        
                        # Capture crop
                        yolo_snap = frame[y1c:y2c, x1c:x2c].copy()
                        name, distances, aligned_face = self.recognizer.identify(yolo_snap)
                        
                        self.active_targets[track_id] = name
                        update_log = f"ID {track_id} LATCHED: {name}"
                        
                        # Fill the package with both images
                        image_package = [yolo_snap, aligned_face]
                    
                    # --- POSSIBILITY 2: Already Tracking (Send frame, [yolo, empty]) ---
                    else:
                        # We still need to draw the box, but we don't update the snaps, image_package remains [empty_img, empty_img]
                        pass

                    # Draw HUD on the main frame
                    if is_enemy and self.is_firing:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.line(frame, (cx-20, cy), (cx+20, cy), (0, 0, 255), 2)
                        cv2.line(frame, (cx, cy-20), (cx, cy+20), (0, 0, 255), 2)

                    name = self.active_targets.get(track_id, "Scanning...")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)
                    cv2.putText(frame, f"{name} (ID:{track_id})", (x1 + 5, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 3. TELEMETRY & EMIT

            # Consistent data package structure
            metadata = {
                "fps": round(fps, 1),
                "update": update_log,
            }

            # Pack the dictionaries
            data_package = [metadata, distances]

            if not self.is_frozen:
                # This signal is now perfectly consistent in its types
                self.update_signal.emit(frame, image_package, data_package)

            # FPS Ceiling Logic, prevents UI from overloading
            processing_time = time.time() - loop_start
            sleep_duration = max(1, int((0.0333 - processing_time) * 1000)) # 0.0333 seconds = 30 FPS
        
        self.msleep(sleep_duration)

    def toggle_freeze(self):
        """Handles the pause/resume logic internally"""
        self.is_frozen = not self.is_frozen
        # Returns the state so the HUD knows what color to turn
        return self.is_frozen

    def reset_tracking_data(self):
        """Clears all identified targets and active memory"""
        self.active_targets.clear()
        self.locked_target_id = None
        self.firing_active = False
        log("SYSTEM REBOOT: Tracking memory cleared.", "INFO")

    def step_forward(self):
        print("I stepped forward")

    def step_backward(self):
        print("I stepped backward")

    def reset_tracking_data(self):
        print("I reset")

    def toggle_fire(self):
        """Logic to switch firing state"""
        self.firing_active = not self.firing_active
        log(f"FIRE STATE: {self.firing_active}", "WARNING")

    def release_target(self):
        """Logic to clear the current lock"""
        self.locked_target_id = None
        self.firing_active = False
        log("TARGET RELEASED - SEARCHING...", "INFO")
