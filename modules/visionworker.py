# modules/interface.py

##################################### Imports #####################################
# Libraries
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
import os
from collections import deque

# Modules
import config
from modules.utils import log, create_event
from modules.detector import YOLODetector, RetinaDetector, SCRFDDetector
from modules.tracker import BoTSORTTracker, ByteTrackTracker
from modules.recognizer import TurretRecognizer
from modules.controller import TurretController

###################################################################################

class VisionWorker(QThread):
    # Signals to communicate with the UI
    # Sends: [Main Frame, Yolo Crop, Data Dict]
    update_signal = pyqtSignal(np.ndarray, list, list)

    def __init__(self, camera_instance):
        super().__init__()
        self.cam = camera_instance # Use the pre-started camera
        self.detector = SCRFDDetector() # RetinaDetector, SCRFDDetector, YOLODetector
        self.tracker = BoTSORTTracker() # ByteTrackTracker
        self.recognizer = TurretRecognizer()
        self.controller = TurretController(simulation=True)

        self.active_targets = {}

        self.box_window_size = 6 # Tuning: Higher = Smoother, but more lag
        self.box_history = {}    # {track_id: deque(maxlen=6)}

        self.running = True
        self.is_frozen = False
        self.is_locking = False
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
            frame_events = [] # logging purposes
            
            # 1. Capture
            frame = self.cam.read()
            clean_frame = frame.copy() # same frame without drawings for UI, I will pass this to AI
            if frame is None or frame.size == 0:
                self.msleep(10); continue
        
            # 2. Scan
            if not self.is_frozen:
                # Step A: Get raw [x1, y1, x2, y2, conf], and facial landmarks from detector
                raw_boxes, landmarks = self.detector.detect(clean_frame)
                
                # Step B: Get [{'id': 1, 'face_bbox': [...], 'center': (...) }] from tracker
                detections = self.tracker.update(raw_boxes, clean_frame)
                
                # Delete IDs from memory that Detector no longer sees in this frame
                current_ids = [d["id"] for d in detections]
                for track_id in list(self.active_targets.keys()):
                    if track_id not in current_ids:
                        del self.active_targets[track_id]
                        del self.box_history[track_id]

                        # If our 'Enemy' is no longer in the active targets, release the lock
                        if track_id == self.locked_target_id:
                            self.locked_target_id = None
                            self.is_firing = False
                            log(f"TARGET LOST: ID {track_id} removed from active memory.", "INFO")

                for target in detections:
                    track_id = target["id"]
                    new_box = np.array(target["face_bbox"], dtype=float)

                    if track_id not in self.box_history:
                        self.box_history[track_id] = deque(maxlen=self.box_window_size)
                    
                    # Add the current detection to the sliding window
                    self.box_history[track_id].append(new_box)

                    # CALCULATE MOVING AVERAGE
                    smoothed_box = np.mean(self.box_history[track_id], axis=0).astype(int)
                    sx1, sy1, sx2, sy2 = smoothed_box

                    target["face_bbox"] = [sx1, sy1, sx2, sy2]
                    target["center"] = ((sx1 + sx2) // 2, (sy1 + sy2) // 2)
                    stx, sty = target["center"][0], target["center"][1]

                    lm_idx = np.argmin([np.linalg.norm(np.array([stx, sty]) - np.mean(lm, axis=0)) for lm in landmarks]) # tracker messes the order so I recheck the get the correct one
                    face_landmarks = landmarks[lm_idx]

                    current_time = time.time()

                    # --- RECOGNITION LOGIC ---
                    # Case A: Brand New Target
                    is_new = track_id not in self.active_targets
                    
                    # Case B: Already Known, but 'Unknown' and 5 seconds have passed
                    should_retry = False
                    if not is_new:
                        target_data = self.active_targets[track_id]
                        if target_data["name"] == "Unknown":
                            last_attempt = target_data.get("last_auth", 0)
                            if (current_time - last_attempt) > 5.0:
                                should_retry = True

                    # --- POSSIBILITY 2: Brand New Target (Send frame, [crop, aligned]) ---
                    if is_new or should_retry:
                        h, w = frame.shape[:2]
                        x1c, y1c, x2c, y2c = max(0, sx1), max(0, sy1), min(w, sx2), min(h, sy2) # border protection
                        detector_crop = clean_frame[y1c:y2c, x1c:x2c].copy()
                        
                        # Recognition
                        name, distances, aligned_face = self.recognizer.identify(clean_frame, face_landmarks)
                        if aligned_face is None or aligned_face.size == 0: continue

                        # Sending format
                        image_package = [detector_crop, aligned_face]
                        
                        self.active_targets[track_id] = {"name": name, "last_auth": current_time}

                        best_filename = sorted(distances.items(), key=lambda x: x[1])[0][0]
                        person_dir = best_filename.rsplit("_", 1)[0]
                        ref_path = os.path.join("assets", "faces", "debug_aligned", person_dir, f"aligned_{best_filename}")
                        frame_events.append(create_event("RECOGNITION", track_id=track_id, name=name, distances=distances, ref_path=ref_path))

                    # --- POSSIBILITY 3: Already Tracking (Send frame, [crop, empty]) ---
                    else:
                        # We still need to draw the box, but we don't update the snaps, image_package remains [empty_img, empty_img]
                        pass
                    
                    name = self.active_targets[track_id]["name"]
                    if name != "Unknown" and self.locked_target_id is None and self.is_locking and affiliation == "ENEMY":
                        self.locked_target_id = track_id
                        frame_events.append(create_event("LOCK", track_id=track_id, status="LOCKED"))

                    # STATUS DETERMINATION (Must happen for everyone)
                    # 1. Determine Affiliation
                    if name in config.ENEMIES:
                        affiliation = "ENEMY"
                        color = config.COLOR_ENEMY
                    elif name in config.FRIENDS:
                        affiliation = "FRIEND"
                        color = config.COLOR_FRIEND
                    else:
                        affiliation = "STRANGER"
                        color = config.COLOR_STRANGER

                    is_enemy = (track_id == self.locked_target_id)
                    thickness = 4 if (is_enemy and self.is_firing) else 2

                    # Draw HUD on the main frame
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
                    cv2.rectangle(frame, (sx1, sy1 - 20), (sx2, sy1), color, -1)

                    display_text = f"{affiliation}: {name} (ID:{track_id})"
                    cv2.putText(frame, display_text, (sx1 + 5, sy1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if is_enemy and self.is_firing:
                        cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                        cv2.line(frame, (cx-20, cy), (cx+20, cy), (0, 0, 255), thickness)
                        cv2.line(frame, (cx, cy-20), (cx, cy+20), (0, 0, 255), thickness)

            # 3. TELEMETRY & EMIT
            delta_time = loop_start - prev_time
            fps = 1.0 / delta_time if delta_time > 0 else 30.0
            prev_time = loop_start

            # Pack the dictionaries
            data_package = [frame_events, round(fps, 1)]

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
        """ Stop the AI """
        self.is_frozen = not self.is_frozen

        if not self.is_frozen:
            log("AI RESUMED", "INFO")
        return self.is_frozen

    def reset_tracking_data(self):
        """ Clears all identified targets and active memory """
        self.active_targets.clear()
        self.locked_target_id = None
        self.is_firing = False
        log("SYSTEM REBOOT: Tracking memory cleared.", "INFO")

    def switch_target(self, step=1):
        """Cycles the locked_target_id only through ENEMY targets"""
        # 1. Filter active IDs to find only confirmed ENEMIES
        enemy_ids = [
            tid for tid, data in self.active_targets.items() 
            if data["name"] in config.ENEMIES
        ]

        if not enemy_ids:
            log("SWITCH REJECTED: No enemy targets in memory.", "WARNING")
            self.locked_target_id = None
            return None

        try:
            # 2. If already locked on an enemy, find the next one in the list
            if self.locked_target_id in enemy_ids:
                current_idx = enemy_ids.index(self.locked_target_id)
                next_idx = (current_idx + step) % len(enemy_ids)
                self.locked_target_id = enemy_ids[next_idx]
            else:
                # 3. If lock was lost or on a non-enemy, grab the first available enemy
                self.locked_target_id = enemy_ids[0]

            log(f"SWITCHED: Locked onto ENEMY ID {self.locked_target_id}", "WARNING")

        except Exception as e:
            log(f"Switch Error: {e}", "ERROR")
            return None

        return self.locked_target_id
    
    def toggle_lock(self):
        """Toggle Active Tracking (Latches ONLY onto Enemies)"""
        self.is_locking = not self.is_locking

        if not self.is_locking:
            self.locked_target_id = None
            self.is_firing = False
            log("TURRET: Lock Revoked. Returning to Overwatch.", "INFO")
        else:
            # Filter for enemies only
            enemy_ids = [
                tid for tid, data in self.active_targets.items() 
                if data["name"] in config.ENEMIES
            ]
            
            if enemy_ids:
                self.locked_target_id = enemy_ids[0]
                log(f"TURRET: Lock Requested. Latching to ENEMY ID {self.locked_target_id}", "WARNING")
            else:
                self.locked_target_id = None
                log("TURRET: Lock Requested. No ENEMIES in sight, standing by...", "WARNING")
        
        return self.is_locking

    def trigger_fire(self):
        """Master trigger: Only works if locked target is an ENEMY"""
        if self.locked_target_id is not None:
            # Double-check affiliation before pulling the trigger
            target_data = self.active_targets.get(self.locked_target_id)
            if target_data and target_data["name"] in config.ENEMIES:
                self.is_firing = not self.is_firing
                status = "FIRE" if self.is_firing else "CEASE FIRE"
                log(f"WEAPON SYSTEM: {status}", "WARNING")
            else:
                self.is_firing = False
                log("FIRE REJECTED: Current lock is NOT an enemy!", "ERROR")
        else:
            self.is_firing = False
            log("FIRE REJECTED: System requires active lock.", "ERROR")
            
        return self.is_firing
    
    ###################################################################################
    #                              CONTROLLER EMIT
    ###################################################################################

    def transmit_to_controller(self, pan_error, tilt_error, fire_command):
        """
        Placeholder for PLC/Microcontroller communication.
        pan_error: float (-1.0 to 1.0)
        tilt_error: float (-1.0 to 1.0)
        fire_command: bool
        """

        pass

    
