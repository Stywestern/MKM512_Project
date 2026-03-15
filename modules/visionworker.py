# modules/interface.py

##################################### Imports #####################################
# Standart Libraries
import cv2
import time
import os

# Third Party Libraries
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
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
    # Sends: [Main Frame, Detect Crop, Data Dict]
    update_signal = pyqtSignal(np.ndarray, list, list)

    def __init__(self, camera_instance):
        super().__init__()
        self.cam = camera_instance # Use the pre-started camera
        self.detector = SCRFDDetector() # RetinaDetector, SCRFDDetector, YOLODetector
        self.tracker = BoTSORTTracker() # ByteTrackTracker
        self.recognizer = TurretRecognizer()
        self.controller = TurretController(simulation=True)

        self.prev_time = 0
        self.active_targets = {}
        self.box_window_size = 6 # Tuning: Higher = Smoother, but more lag
        self.box_history = {}    # {track_id: deque(maxlen=6)}

        self.running = True
        self.is_frozen = False
        self.is_locking = False
        self.locked_target_id = None  # ID of the current "Enemy"
        self.is_firing = False

        log("VisionWorker initialized", "INFO")

    ###################################################################################
    #                                 HELPER METHODS
    ###################################################################################

    def _purge_stale_targets(self, current_ids):
        """
        Cleans up memory for targets no longer detected in the current frame.
        Ensures Weapon Safety by revoking locks on lost targets.
        """
        # Create a list of IDs to remove to avoid 
        targets_to_remove = [tid for tid in self.active_targets if tid not in current_ids]

        for tid in targets_to_remove:
            # 1. Clear Identity and Distance Memory
            if tid in self.active_targets:
                del self.active_targets[tid]
            
            # 2. Clear Smoothing/Jitter Buffers
            if tid in self.box_history:
                del self.box_history[tid]
                
            # 3. If locked, release the system
            if tid == self.locked_target_id:
                self.locked_target_id = None
                self.is_firing = False
                log(f"TARGET LOST: ID {tid} removed. System returning to Overwatch.", "INFO")
            else:
                log(f"Memory Cleared: ID {tid} (Stale)", "DEBUG")


    def _apply_temporal_smoothing(self, target):
        """
        Filters high-frequency jitter using a Moving Average buffer.
        Updates the target's bbox and center coordinates.
        """

        tid = target["id"]
        raw_box = np.array(target["face_bbox"], dtype=float)

        # Initialize buffer if new ID
        if tid not in self.box_history:
            self.box_history[tid] = deque(maxlen=self.box_window_size)
        
        self.box_history[tid].append(raw_box)
        
        # Calculate Mean
        smoothed = np.mean(self.box_history[tid], axis=0).astype(int)
        
        # Update object
        target["face_bbox"] = [smoothed[0], smoothed[1], smoothed[2], smoothed[3]]
        target["center"] = ((smoothed[0] + smoothed[2]) // 2, 
                            (smoothed[1] + smoothed[3]) // 2)
        

    def _sync_sensors_to_target(self, target, landmarks, raw_distances):
        """
        Finds the closest raw detection landmarks for a tracked ID. 
        I did this because detector -> tracker pass sometimes messes with the ordering.
        """
        scx, scy = target["center"]
        
        # Spatial Matching: Find the raw landmark set closest to smoothed center
        lm_idx = np.argmin([
            np.linalg.norm(np.array([scx, scy]) - np.mean(lm, axis=0)) 
            for lm in landmarks
        ])
        
        # Update Distance Latch
        current_dist = raw_distances[lm_idx] if lm_idx < len(raw_distances) else None

        if current_dist is not None:
            try:
                self.active_targets[target["id"]]["distance"] = current_dist
            except:
                pass

        return current_dist, landmarks[lm_idx]
    

    def _should_identify(self, track_id):
        """
        Determines if a specific target requires a fresh recognition attempt.
        Currently triggers if the target is 'Unknown' and 5 seconds have passed.
        """
        current_time = time.time()
        target_data = self.active_targets.get(track_id)

        # 1. If we don't have this ID in memory at all, it's a 'New' target
        if not target_data:
            return True

        # 2. Logic for 'Unknown' targets
        if target_data.get("name") == "Unknown":
            last_attempt = target_data.get("last_auth", 0)
            
            # 5-second cooldown to prevent spamming the Embedding model
            if (current_time - last_attempt) > 5.0:
                return True

        # 3. Future Expansion: Add rules for 'Low Confidence' or 'Distance Changes'
        return False
    
    def _arbitrate_target_lock(self, potential_enemies):
        """
        Decides which target to lock onto if no lock currently exists.
        Can be expanded to include distance or priority-based sorting.
        """

        # 1. Early exit: If we aren't in locking mode or already have a lock
        if not self.is_locking or self.locked_target_id is not None:
            return None

        # 2. Early exit: No enemies present
        if not potential_enemies:
            return None

        # 3. SORTING LOGIC (The 'Doctrine')
        potential_enemies.sort(key=lambda x: self.active_targets.get(x["id"], {}).get("distance", 200.0))

        # 4. SELECT AND LOCK
        best_target = potential_enemies[0]
        self.locked_target_id = best_target["id"]
        
        log(f"TACTICAL ARBITRATOR: Locked onto ID {self.locked_target_id} (Closest Enemy)", "WARNING")
        
        # Return an event to be added to the UI logs
        return create_event("LOCK", track_id=self.locked_target_id, status="LOCKED")
    
    def _draw_target_hud(self, frame, target, name, affiliation, color, distance):
        """
        Handles all visual overlays for a single target.
        Logic:
        1. Draw the bounding box and header bar.
        2. Overlay telemetry (Name, ID, Distance).
        3. If currently firing at THIS target, draw the red engagement crosshair.
        """
        sx1, sy1, sx2, sy2 = target["face_bbox"]
        track_id = target["id"]
        
        # 1. Determine if this is the ACTIVE engagement target
        is_locked_target = (track_id == self.locked_target_id)
        is_actively_firing = (is_locked_target and self.is_firing)
        
        # Visual thickness increases when firing for 'recoil' effect
        thickness = 4 if is_actively_firing else 2

        # 2. Draw Bounding Box & Identity Header
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.rectangle(frame, (sx1, sy1 - 22), (sx2, sy1), color, -1)

        # 3. Telemetry String
        # Format: ENEMY: Kerem (ID:5)(DIST: 150.2cm)
        display_text = f"{affiliation}: {name} (ID:{track_id})(DIST: {distance:.1f}cm)"
        
        cv2.putText(frame, display_text, (sx1 + 5, sy1 - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Engagement Crosshair (Only if firing)
        if is_actively_firing:
            cx, cy = target["center"]
            # Red crosshair centered on the smoothed face center
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 0, 255), thickness)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 0, 255), thickness)
            # Optional: Add a 'FIRE' alert next to the box
            cv2.putText(frame, "ENGAGING", (sx1, sy2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    def _finalize_cycle(self, frame, image_package, frame_events, loop_start):
        """
        Handles telemetry calculation, UI communication, and thread timing.
        Ensures the loop maintains a stable framerate.
        """
        # 1. Calculate FPS
        # We use self.prev_time (stored in the class) to calculate the delta
        current_time = time.time()
        delta_time = current_time - self.prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 30.0
        self.prev_time = current_time

        # 2. Package and Emit to UI
        data_package = [frame_events, round(fps, 1)]
        self.update_signal.emit(frame, image_package, data_package)

        # 3. Dynamic Sleep (FPS Governor)
        # Target: 33.3ms per frame (approx 30 FPS)
        processing_time = time.time() - loop_start
        target_period = 0.0333 
        
        sleep_duration = max(1, int((target_period - processing_time) * 1000))
        self.msleep(sleep_duration)

    ###################################################################################
    #                                 MAIN LOOP
    ###################################################################################

    def run(self):
        self.prev_time = time.time()
        log("Running Sentry Logic Subsystem", "INFO")
        
        while self.running:
            loop_start = time.time()
            
            # POSSIBILITY 1: No Face Detected (Just send the frame) 
            empty_img = np.array([], dtype=np.uint8)
            image_package = [empty_img, empty_img] # [YOLO_CROP, ALIGN_CROP]
            frame_events = [] # logging purposes
            
            # 1. Capture the frame
            frame = self.cam.read()
            clean_frame = frame.copy() # same frame without drawings for UI, I will pass this to AI
            if frame is None or frame.size == 0: self.msleep(10); continue
        
            # 2. Scan for detection
            if not self.is_frozen:

                # Step A: Get raw [x1, y1, x2, y2, conf], and facial landmarks from detector
                raw_boxes, landmarks, raw_distances = self.detector.detect(clean_frame)
                
                # Step B: Get [{'id': 1, 'face_bbox': [...], 'center': (...) }] from tracker
                detections = self.tracker.update(raw_boxes, clean_frame)
                
                # Step B.1.: Purge ids that are absent from the frame
                current_ids = [d["id"] for d in detections]
                self._purge_stale_targets(current_ids)

# --------------------------------- Step C (Starts): Start loop for one target ----------------------------------------
                potential_enemies = []

                for target in detections:
                    # ------------------- PREPROCESSING (START) ---------------

                    self._apply_temporal_smoothing(target) # smoothens the box
                    current_dist, face_landmarks = self._sync_sensors_to_target(target, landmarks, raw_distances) # returns correct landmarks

                    track_id = target["id"]
                    sx1, sy1, sx2, sy2 = target["face_bbox"]

                    # ------------------- PREPROCESSING (END) ---------------

                    # -------------- RECOGNITION (START) -----------------------

                    # POSSIBILITY 2: Brand New Target (Send frame, [crop, aligned])
                    current_time = time.time()
                    if self._should_identify(track_id):

                        # C.1. Crop the correct frame
                        h, w = frame.shape[:2]
                        x1c, y1c, x2c, y2c = max(0, sx1), max(0, sy1), min(w, sx2), min(h, sy2)
                        detector_crop = clean_frame[y1c:y2c, x1c:x2c].copy()
                        
                        # C.2. Run recognition, returns a name, scores dict, aligned_face image for debug
                        name, distances, aligned_face = self.recognizer.identify(clean_frame, face_landmarks)
                        if aligned_face is None or aligned_face.size == 0: continue

                        # C.3. Update emittion data
                        image_package = [detector_crop, aligned_face]
                        
                        self.active_targets[track_id] = {"name": name, "last_auth": current_time, "distance": current_dist or 200.0}

                        best_filename = sorted(distances.items(), key=lambda x: x[1])[0][0]
                        person_dir = best_filename.rsplit("_", 1)[0]
                        ref_path = os.path.join("assets", "faces", "debug_aligned", person_dir, f"aligned_{best_filename}")
                        frame_events.append(create_event("RECOGNITION", track_id=track_id, name=name, distances=distances, ref_path=ref_path))

                    # POSSIBILITY 3: Already Tracking (Send frame, [crop, empty])
                    else:
                        # We still need to draw the box, but we don't update the snaps, image_package remains [empty_img, empty_img], we pass the stuff as it is
                        if current_dist is not None:
                            self.active_targets[track_id]["distance"] = current_dist

                        name = self.active_targets[track_id]["name"]

                    # C.4. Determine Affiliation
                    if name in config.ENEMIES:
                        affiliation = "ENEMY"
                        color = config.COLOR_ENEMY
                        potential_enemies.append(target)

                    elif name in config.FRIENDS:
                        affiliation = "FRIEND"
                        color = config.COLOR_FRIEND
                    else:
                        affiliation = "STRANGER"
                        color = config.COLOR_STRANGER

                    # -------------- RECOGNITION (END) -----------------------

                    # -------------- VISUALIZATION (START) ----------------------- 
                    self._draw_target_hud(frame, target, name, affiliation, color, current_dist or 200.0)

                    # -------------- VISUALIZATION (END) ----------------------- 

# --------------------------------- Step C (Ends): End loop for one target ----------------------------------------

            # 3. TELEMETRY & EMIT

            # A. Check if locking is going on
            lock_event = self._arbitrate_target_lock(potential_enemies)
            if lock_event:
                frame_events.append(lock_event)

            # B. Send data to the PLC
            if self.locked_target_id is not None:
                # 1. Find the target dictionary in the CURRENT detections list
                # We need the current frame's center (scx, scy)
                locked_target_obj = next((d for d in detections if d["id"] == self.locked_target_id), None)
                
                if locked_target_obj:
                    # 2. Calculate vector (Using our Parallax math)
                    pan_err, tilt_err = self._calculate_targeting_vector(locked_target_obj)
                    print(pan_err, tilt_err)
                    
                    # 3. Fire Command (Only fire if they are an ENEMY and we are in firing mode)
                    # Note: We already checked they were an enemy to lock them
                    self.controller.update_turret(pan_err, tilt_err, self.is_firing)
                else:
                    # Target is gone! Purge will handle memory, but we must stop motors now.
                    self.controller.update_turret(0, 0, False)
            else:
                # No lock? Standby.
                self.controller.update_turret(0, 0, False)

            # C. Send the loop info
            self._finalize_cycle(frame, image_package, frame_events, loop_start)
        
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

    def _calculate_targeting_vector(self, target):
        """
        Translates pixel coordinates and distance into physical angles.
        Includes Parallax Correction for the camera-to-barrel offset.
        """
        cx, cy = target["center"]
        dist_cm = self.active_targets[target["id"]].get("distance", 200.0)

        # 1. Get Pixel Error from Screen Center (640, 360)
        dx = cx - 640
        dy = cy - 360 # Note: In pixels, Y increases downwards

        # 2. Convert Pixels to Radians (using our Focal Length)
        # Formula: theta = arctan(pixels / focal_length)
        yaw_rad = np.arctan2(dx, config.FOCAL_LENGTH)
        pitch_rad = np.arctan2(dy, config.FOCAL_LENGTH)

        # 3. PARALLAX CORRECTION (Vertical Offset)
        # Assume camera is 10cm ABOVE the barrel
        camera_offset_y = 10.0 
        # At 'dist_cm', the barrel needs to tilt UP slightly more than the camera sees
        # correction_angle = arctan(offset / distance)
        parallax_correction = np.arctan2(camera_offset_y, dist_cm)
        
        # Final Pitch = Visual Pitch + Parallax Correction
        corrected_pitch_rad = pitch_rad + parallax_correction

        # 4. Convert to normalized units (-1.0 to 1.0) for the Comms module
        # We assume our "Field of View" is the limit
        pan_error = np.degrees(yaw_rad) / 30.0   # Normalized to a 60deg total span
        tilt_error = np.degrees(corrected_pitch_rad) / 20.0 

        return np.clip(pan_error, -1.0, 1.0), np.clip(tilt_error, -1.0, 1.0)


    def transmit_to_controller(self, pan_error, tilt_error, fire_command):
        """
        Placeholder for PLC/Microcontroller communication.
        pan_error: float (-1.0 to 1.0)
        tilt_error: float (-1.0 to 1.0)
        fire_command: bool
        """

        pass



    
