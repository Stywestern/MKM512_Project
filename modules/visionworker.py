# modules/visionworker.py

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
from modules.controller import RealTurretController, SimTurretController
from modules.PLC import TurretPLC

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

        self.plc = TurretPLC(ip="192.168.0.101", port=23000)
        if self.plc.connect():
            log("HARDWARE: PLC Connected. Physical turret ACTIVE.", "INFO")
            #self.controller = RealTurretController(self.plc)
            self.controller = RealTurretController(self.plc)
        else:
            log("HARDWARE: PLC Offline. Running in SIMULATION MODE.", "WARNING")
            self.controller = SimTurretController()


        self.prev_time = 0
        self.active_targets = {}
        self.box_window_size = 2 # Tuning: Higher = Smoother, but more lag
        self.box_history = {}    # {track_id: deque(maxlen=6)}

        self.running = True
        self.is_frozen = True
        self.is_locking = False
        self.locked_target_id = None  # ID of the current "Enemy"
        self.is_firing = False

        log("VisionWorker initialized", "INFO")

    def shutdown(self):
        """ Deterministic teardown method. Ensures hardware returns to safe state. """
        log("SYSTEM SHUTDOWN: Initiating safe teardown...", "WARNING")
        
        # 1. Stop the AI loop
        self.running = False 
        
        # 2. Tell the hardware controller thread to die gracefully
        if hasattr(self, 'controller') and hasattr(self.controller, 'hw_thread'):
            self.controller.hw_thread.stop()  # Sets hw_thread.running = False
            
            self.controller.hw_thread.join(timeout=2.0) 
            
        # 3. Stop the camera hardware
        if hasattr(self, 'cam'):
            self.cam.stop()

    ###################################################################################
    #                                 HELPER METHODS
    ###################################################################################

    def _purge_stale_targets(self, current_ids):
        """
        Cleans up memory for targets no longer detected.
        Uses a 2.0 second grace period to survive micro-stutters and tracker buffering.
        """
        current_time = time.time()
        targets_to_remove = []
        
        # Determine who has been missing for too long
        for tid, data in self.active_targets.items():
            if tid not in current_ids:
                # Target is missing this frame, check the grace period
                time_lost = current_time - data.get("last_seen", current_time)
                
                # If lost for more than 2 seconds, mark for permanent deletion
                if time_lost > 2.0:
                    targets_to_remove.append(tid)

        # Execute the purge only on truly dead targets
        for tid in targets_to_remove:
            # 1. Clear Identity and Distance Memory
            if tid in self.active_targets:
                del self.active_targets[tid]
            
            # 2. Clear Smoothing/Jitter Buffers
            if tid in self.box_history:
                del self.box_history[tid]
                
            # 3. Drop the lock so the Arbitrator can acquire new targets
            if tid == self.locked_target_id:
                self.locked_target_id = None
                #self.is_firing = False
                log(f"TARGET LOST: ID {tid} completely removed. Lock released.", "WARNING")

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
        

    def _sync_sensors_to_target(self, target, landmarks):
        """ 
        Finds the closest raw detection landmarks for a tracked ID 
        and calculates true PnP distance.
        """
        if landmarks is None or len(landmarks) == 0:
            return None, []

        scx, scy = target["center"]
        
        try:
            # Spatial Matching: Find the correct landmarks for this bounding box
            lm_idx = np.argmin([
                np.linalg.norm(np.array([scx, scy]) - np.mean(lm, axis=0)) 
                for lm in landmarks
            ])
            
            target_landmarks = landmarks[lm_idx]
            
            # PnP
            current_dist = self._estimate_distance_pnp(target_landmarks)

            # Safe Dictionary Assignment
            if current_dist is not None:
                if target["id"] not in self.active_targets:
                    self.active_targets[target["id"]] = {}
                self.active_targets[target["id"]]["distance"] = current_dist

            return current_dist, target_landmarks
            
        except Exception as e:
            log(f"Sensor Sync Error: {e} - Skipping frame sync.", "WARNING")
            return None, []

    def _should_identify(self, track_id):
        """
        Determines if a specific target requires a fresh recognition attempt.
        """
        current_time = time.time()
        target_data = self.active_targets.get(track_id)

        # If we don't have this ID in memory at all, OR it exists but lacks a name, it's a 'New' target
        if not target_data or "name" not in target_data:
            return True

        # Logic for 'Unknown' targets (Retries every 5 seconds)
        if target_data.get("name") == "Unknown":
            last_attempt = target_data.get("last_auth", 0)
            if (current_time - last_attempt) > 5.0:
                return True

        return False
    
    def _arbitrate_target_lock(self, potential_enemies):
        """ Decides which target to lock onto if no lock currently exists. """
        if not self.is_locking or self.locked_target_id is not None:
            return None

        if not potential_enemies:
            return None

        try:
            # 1. FAULT TOLERANCE: Force cast to float, and catch NoneTypes with 'or 200.0'
            potential_enemies.sort(
                key=lambda x: float(self.active_targets.get(x["id"], {}).get("distance") or 200.0)
            )

            # 2. SELECT AND LOCK
            best_target = potential_enemies[0]
            self.locked_target_id = best_target["id"]
            
            log(f"TACTICAL ARBITRATOR: Locked onto ID {self.locked_target_id} (Closest Enemy)", "WARNING")
            return create_event("LOCK", track_id=self.locked_target_id, status="LOCKED")
            
        except Exception as e:
            log(f"Arbitrator Sorting Error: {e}", "ERROR")
            return None
    
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
        is_firing = (is_locked_target and self.is_firing)
        
        thickness = 4 if is_firing else 2

        # 2. Format the Display Text
        first_name = name.replace("_", " ").split(" ")[0]
        affil_char = affiliation[0] 
        display_text = f"[{affil_char}] {first_name} #{track_id} | d:{int(distance)}cm"

        # 3. Upgraded Text Sizing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65  # Increased from 0.5 for much better readability
        font_thickness = 2 
        
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)

        # --- THE FIX: FULL SPAN WIDTH ---
        # The box must be AT LEAST the width of the face, but expand if the text is longer.
        face_width = sx2 - sx1
        box_width = max(text_width + 20, face_width) 
        
        # 4. Draw Bounding Box
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, thickness)
        
        # 5. Draw Full-Span Header Background
        header_y1 = max(0, sy1 - text_height - 15) # Added vertical padding
        cv2.rectangle(frame, (sx1, header_y1), (sx1 + box_width, sy1), color, -1)

        # --- THE FIX: PERFECT CENTERING ---
        # Calculate exactly where the text should start to be centered in the box
        text_x = sx1 + (box_width - text_width) // 2
        text_y = sy1 - 7 

        # 6. Draw High-Contrast Bold Black Text
        cv2.putText(frame, display_text, (text_x, text_y), 
                    font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # 7. Engagement Crosshair & Alert
        if is_firing:
            cx, cy = target["center"]
            
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 0, 255), thickness)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 0, 255), thickness)
            
            (e_w, e_h), _ = cv2.getTextSize("ENGAGING", font, 0.7, 2)
            
            # Center the "ENGAGING" alert below the box as well
            alert_x = sx1 + (face_width - e_w) // 2
            cv2.rectangle(frame, (alert_x - 5, sy2), (alert_x + e_w + 5, sy2 + e_h + 10), (0, 0, 255), -1)
            cv2.putText(frame, "ENGAGING", (alert_x, sy2 + e_h + 5), 
                        font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            
    def _estimate_distance_pnp(self, landmarks):
        """
        Uses Perspective-n-Point (PnP) to calculate true 3D distance in cm.
        Calibrated for A4Tech PK-910H (1080p, 70-deg FOV) with Nose-Origin.
        """
        try:
            if landmarks is None or len(landmarks) != 5:
                return None

            # 1. Generic 3D Adult Face Model (in cm). Origin is Nose Tip.
            model_points = np.array([
                [-3.4, -3.0,  3.0],  # Left Eye 
                [ 3.4, -3.0,  3.0],  # Right Eye
                [ 0.0,  0.0,  0.0],  # Nose Tip (Origin)
                [-2.6,  4.0,  3.0],  # Left Mouth
                [ 2.6,  4.0,  3.0]   # Right Mouth
            ], dtype=np.float32)

            image_points = np.array(landmarks, dtype=np.float32)

            # 2. A4Tech PK-910H Intrinsics
            focal_length = config.FOCAL_LENGTH
            center_x = config.FRAME_WIDTH / 2.0
            center_y = config.FRAME_HEIGHT / 2.0
            
            camera_matrix = np.array([
                [focal_length, 0.0, center_x],
                [0.0, focal_length, center_y],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))

            # 3. Solve PnP (SQPNP for 5-point stability)
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
            )

            if success:
                # Deep index [2][0] to prevent the scalar NumPy crash
                z_distance_cm = float(translation_vec[2][0]) 
                if 10.0 < z_distance_cm < 1000.0:
                    return z_distance_cm

            return None
            
        except Exception as e:
            log(f"PnP Math Error: {e}", "WARNING")
            return None

    def _draw_kinematic_debug(self, frame, target, x_diff, y_diff):
        """
        Draws visual markers to debug the pixel-to-motor coordinate mapping.
        """
        cx, cy = target["center"]
        
        # The assumed physical center of your rotated frame
        center_x = config.FRAME_WIDTH // 2
        center_y = config.FRAME_HEIGHT // 2
        
        # 1. Draw the assumed screen center (Blue Dot)
        cv2.circle(frame, (center_x, center_y), 6, (255, 0, 0), -1)
        cv2.putText(frame, "CENTER", (center_x + 10, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 2. Draw the face center (Green Dot)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        
        # 3. Draw the error vector line (Yellow Line)
        cv2.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 2)

        # 4. Display the normalized math being sent to the controller
        debug_text = f"Nx: {x_diff:+.2f} | Ny: {y_diff:+.2f}"
        cv2.putText(frame, debug_text, (cx + 10, cy + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    def _finalize_cycle(self, frame, image_package, frame_events, loop_start):
        current_time = time.time()
        delta_time = current_time - self.prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 30.0
        self.prev_time = current_time

        system_state = {
            "is_locking": self.is_locking,                     
            "has_target": self.locked_target_id is not None,   
            "is_firing": self.is_firing
        }

        # Append system_state as the 3rd item in the data package
        data_package = [frame_events, round(fps, 1), system_state]
        self.update_signal.emit(frame, image_package, data_package)

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

        detections = []
        potential_enemies = []

        # Boot turret 
        if self.plc.connected:
            log("BOOT: Initializing Overwatch", "INFO")
            # Set the initial position and dynamics 
            self.plc.send_pose(0, 0)
            #self.plc.set_laser(False)

            self.plc.set_velocity(tilt_vel=500, pan_vel=50)

        while self.running:
            loop_start = time.time()
            
            # POSSIBILITY 1: No Face Detected (Just send the frame) 
            empty_img = np.array([], dtype=np.uint8)
            image_package = [empty_img, empty_img] # [YOLO_CROP, ALIGN_CROP]
            frame_events = [] # logging purposes
            
            # 1. Capture the frame
            frame = self.cam.read()
            pristine_frame = frame.copy() # PRISTINE: Untouched. Used strictly for pure image crops.
            ai_frame = frame.copy() # AI FRAME: Sent to models (in case they secretly mutate arrays)
            display_frame = frame.copy()  # DISPLAY FRAME: The canvas for your HUD and crosshairs

            if frame is None or frame.size == 0: self.msleep(10); continue
        
            # 2. Scan for detection
            try:
                if not self.is_frozen:

                    # Step A: Get raw [x1, y1, x2, y2, conf], and facial landmarks from detector
                    raw_boxes, landmarks, raw_distances = self.detector.detect(ai_frame)
                    
                    # Step B: Get [{'id': 1, 'face_bbox': [...], 'center': (...) }] from tracker
                    detections = self.tracker.update(raw_boxes, ai_frame)
                    
                    # Step B.1.: Purge ids that are absent from the frame
                    current_ids = [d["id"] for d in detections]
                    self._purge_stale_targets(current_ids)

                    # --------------------------------- Step C (Starts): Start loop for one target ----------------------------------------
                    potential_enemies = []

                    for target in detections:
                        # ------------------- PREPROCESSING (START) ---------------

                        self._apply_temporal_smoothing(target) # smoothens the box
                        current_dist, face_landmarks = self._sync_sensors_to_target(target, landmarks) # returns correct landmarks

                        track_id = target["id"]
                        sx1, sy1, sx2, sy2 = target["face_bbox"]

                        # ------------------- PREPROCESSING (END) ---------------

                        # -------------- RECOGNITION (START) -----------------------

                        # POSSIBILITY 2: Brand New Target (Send frame, [crop, aligned])
                        current_time = time.time()
                        if self._should_identify(track_id):

                            # C.1. Crop the correct frame
                            h, w = pristine_frame.shape[:2]
                            x1c, y1c, x2c, y2c = max(0, sx1), max(0, sy1), min(w, sx2), min(h, sy2)
                            detector_crop = pristine_frame[y1c:y2c, x1c:x2c].copy()
                            
                            # C.2. Run recognition, returns a name, scores dict, aligned_face image for debug
                            name, distances, aligned_face = self.recognizer.identify(pristine_frame, face_landmarks)
                            if aligned_face is None or aligned_face.size == 0: continue

                            # C.3. Update emittion data
                            image_package = [detector_crop, aligned_face]
                            
                            self.active_targets[track_id] = {"name": name, "last_auth": current_time, "distance": current_dist or 200.0, "last_seen": current_time}

                            best_filename = sorted(distances.items(), key=lambda x: x[1])[0][0]
                            person_dir = best_filename.rsplit("_", 1)[0]
                            ref_path = os.path.join("assets", "faces", "debug_aligned", person_dir, f"aligned_{best_filename}")
                            
                            frame_events.append(create_event("RECOGNITION", track_id=track_id, name=name, distances=distances, ref_path=ref_path))

                            log(f"New Recognition: {name}", "DEBUG")

                        # POSSIBILITY 3: Already Tracking (Send frame, [crop, empty])
                        else:
                            # We still need to draw the box, but we don't update the snaps, image_package remains [empty_img, empty_img], we pass the stuff as it is
                            if current_dist is not None:
                                self.active_targets[track_id]["distance"] = current_dist

                            self.active_targets[track_id]["last_seen"] = current_time
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
                        self._draw_target_hud(display_frame, target, name, affiliation, color, current_dist or 200.0)

                        pan_err, tilt_err = self._calculate_targeting_vector(target)

                        self._draw_kinematic_debug(display_frame, target, pan_err, tilt_err)

                        # -------------- VISUALIZATION (END) ----------------------- 

                        # --------------------------------- Step C (Ends): End loop for one target ----------------------------------------
            except Exception as e:
                    # If literally anything blows up, print the exact error, but keep the loop alive
                    log(f"FATAL PIPELINE CRASH: {e} - Skipping corrupted frame.", "ERROR")
                    self.controller.update_turret(0.0, 0.0, 200.0, False)
        
            # 3. TELEMETRY & EMIT

            # A. Check if locking is going on

            lock_event = self._arbitrate_target_lock(potential_enemies)
            if lock_event:
                frame_events.append(lock_event)

            # B. Send data to the PLC
            if self.locked_target_id is not None:
                # 1. Find the target dictionary in the CURRENT detections list
                locked_target_obj = next((d for d in detections if d["id"] == self.locked_target_id), None)
                
                if locked_target_obj:
                    # Target is visible
                    pan_err, tilt_err = self._calculate_targeting_vector(locked_target_obj)
                    target_data = self.active_targets.get(self.locked_target_id, {})
                    dist = target_data.get("distance", 200.0)

                    self.controller.update_turret(pan_err, tilt_err, dist, self.is_firing)
                else:
                    # Fetch last known distance so the UI doesn't glitch to 0.0
                    last_dist = self.active_targets.get(self.locked_target_id, {}).get("distance", 200.0)
                    
                    # Update turret with 0.0 error (adds 0 to current pos)
                    self.controller.update_turret(0.0, 0.0, last_dist, False)

            elif self.is_locking == True and self.locked_target_id is None:
                # Hold the ground
                self.controller.update_turret(0.0, 0.0, 0, self.is_firing)

            else:
                # Return to overwatch
                self.controller.perform_overwatch()


            # C. Send the loop info
            #print(self.locked_target_id)
            self._finalize_cycle(display_frame, image_package, frame_events, loop_start)
        
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
        """ Clears all identified targets and active memory safely """
        # 1. First, tell the AI loop to drop the lock and stop firing
        self.locked_target_id = None
        self.is_firing = False
        
        # 2. Then, wipe the memory dictionaries
        self.active_targets.clear()
        self.box_history.clear()
        
        log("SYSTEM REBOOT: Tracking memory cleared.", "INFO")

    def switch_target(self, step=1):
        """Cycles the locked_target_id only through ENEMY targets"""
        # 1. Filter active IDs to find only confirmed ENEMIES
        enemy_ids = [
            tid for tid, data in self.active_targets.items() 
            if data["name"] in config.ENEMIES
        ]

        if not self.is_locking:
            log("SWITCH REJECTED: System is in Overwatch mode.", "WARNING")
            return None
        
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
        Pure Sensor Logic: Returns normalized pixel error [-1.0 to 1.0].
        Assumes an upright portrait orientation (e.g., 720x1280).
        """
        cx, cy = target["center"]
        
        # 1. Update Centers for Portrait Mode
        # If your res is different, change these to (width/2) and (height/2)
        center_x = config.FRAME_WIDTH // 2 
        center_y = config.FRAME_HEIGHT // 2
        
        # 2. Raw Pixel Error
        dx = center_x - cx
        dy = cy - center_y 

        return dx, dy

    def transmit_to_controller(self, pan_error, tilt_error, dist_cm, fire_command):
        """
        Passes targeting data to the controller. 
        If PLC is offline, this call safely does nothing or logs simulation data.
        """
        self.controller.update_turret(pan_error, tilt_error, dist_cm, fire_command)