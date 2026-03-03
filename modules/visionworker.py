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
        self.lock_requested = False
        self.locked_target_id = None  # ID of the current "Enemy"
        self.is_firing = False

        log("VisionWorker initialized", "INFO")

    def run(self):
        prev_time = time.time()
        log("Running Sentry Logic Subsystem", "INFO")
        
        while self.running:
            loop_start = time.time()
            
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
                current_ids = [d["id"] for d in detections]

                # Delete IDs from memory that YOLO no longer sees in this frame
                for track_id in list(self.active_targets.keys()):
                    if track_id not in current_ids:
                        del self.active_targets[track_id]

                        # If our 'Enemy' is no longer in the active targets, release the lock
                        if track_id == self.locked_target_id:
                            self.locked_target_id = None
                            self.is_firing = False
                            log(f"TARGET LOST: ID {track_id} removed from active memory.", "INFO")

                for target in detections:
                    track_id = target["id"]
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

                        if name != "Unknown" and self.locked_target_id is None and self.lock_requested:
                            self.locked_target_id = track_id
                            update_log = f"ID {track_id} is the target"

                        # Fill the package with both images
                        image_package = [yolo_snap, aligned_face]

                        is_enemy = (track_id == self.locked_target_id)
                        color = (0, 0, 255) if is_enemy else (0, 255, 0)
                        thickness = 4 if (is_enemy and self.is_firing) else 2
                    
                    # --- POSSIBILITY 2: Already Tracking (Send frame, [yolo, empty]) ---
                    else:
                        # We still need to draw the box, but we don't update the snaps, image_package remains [empty_img, empty_img]
                        pass

                    # Draw HUD on the main frame
                    if is_enemy and self.is_firing:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.line(frame, (cx-20, cy), (cx+20, cy), (0, 0, 255), thickness)
                        cv2.line(frame, (cx, cy-20), (cx, cy+20), (0, 0, 255), thickness)

                    name = self.active_targets.get(track_id, "Scanning...")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)
                    cv2.putText(frame, f"{name} (ID:{track_id})", (x1 + 5, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 3. TELEMETRY & EMIT
            delta_time = loop_start - prev_time
            fps = 1.0 / delta_time if delta_time > 0 else 30.0
            prev_time = loop_start

            metadata = {
                "fps": round(fps, 1),
                "update": update_log,
            }

            # Pack the dictionaries
            data_package = [metadata, distances]

            # Send the data
            self.update_signal.emit(frame, image_package, data_package)

            # 4. SLEEP (FPS Ceiling)
            processing_time = time.time() - loop_start
            sleep_duration = max(1, int((0.0333 - processing_time) * 1000))
            self.msleep(sleep_duration)
        
    ###################################################################################
    #                                 BUTTON LOGIC
    ###################################################################################

    def toggle_freeze(self):
        self.is_frozen = not self.is_frozen

        if not self.is_frozen:
            log("AI RESUMED", "INFO")
        return self.is_frozen

    def reset_tracking_data(self):
        """Clears all identified targets and active memory"""
        self.active_targets.clear()
        self.locked_target_id = None
        self.is_firing = False
        log("SYSTEM REBOOT: Tracking memory cleared.", "INFO")

    def switch_target(self, step=1):
        """Cycles the locked_target_id through the currently active dictionary keys"""
        if not self.active_targets:
            return None

        # Get the list of IDs currently 'alive' in memory
        active_ids = list(self.active_targets.keys())

        try:
            if self.locked_target_id in active_ids:
                current_idx = active_ids.index(self.locked_target_id)
                next_idx = (current_idx + step) % len(active_ids)
                self.locked_target_id = active_ids[next_idx]
            else:
                # If no lock exists or target disappeared, pick the first available
                self.locked_target_id = active_ids[0]

        except Exception as e:
            log(f"Switch Error: {e}", "ERROR")
            return None

        return self.locked_target_id
    
    def toggle_lock(self):
        """
        Switches between Overwatch (No ID locked) and 
        Active Tracking (Latch onto the first available ID).
        """
        self.lock_requested = not self.lock_requested

        # If the user is revoking permission, we must clear any current lock
        if not self.lock_requested:
            self.locked_target_id = None
            self.is_firing = False
            log("TURRET: Lock Revoked. Returning to Overwatch.", "INFO")
        else:
            self.locked_target_id = self.active_targets[0].key()
            log("TURRET: Lock Requested. Seeking target", "WARNING")
            
        return self.lock_requested

    def trigger_fire(self):
        """Master trigger for engagement simulation"""
        if self.locked_target_id is not None:
            self.is_firing = not self.is_firing
            status = "!!! ENGAGING !!!" if self.is_firing else "CEASE FIRE"
            log(f"WEAPON SYSTEM: {status}", "WARNING")
        else:
            self.is_firing = False
            log("FIRE REJECTED: System requires active lock.", "ERROR")
            
        return self.is_firing

    
