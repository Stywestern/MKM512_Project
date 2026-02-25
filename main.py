#main.py

##################################### Imports #####################################
# Libraries
import cv2
import sys
import time
import json

import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

# Modules
from modules.camera import CameraStream
from modules.detector import YOLODetector, RetinaDetector
from modules.recognizer import TurretRecognizer
import config
from modules.utils import log

###################################################################################

def main():
    log("System starting...", "INFO")

    # 1. Initialize the Camera and start it
    cam = CameraStream(src=config.CAMERA_INDEX).start()
    log(f"Capture from: {cam}", "INFO")

    # 2. Initialize the models and start them 
    detector = YOLODetector(threshold=config.DET_CONF_THRESHOLD)
    log(f"Detector Model: {detector}", "INFO")

    recognizer = TurretRecognizer(model_name="w600k_r50", threshold=config.REG_CONF_THRESHOLD)
    log(f"Recognizer Model: {recognizer}", "INFO")

    # 3. Out of the loop definitions, must be made here to avoid scope issues
    identified_targets = {} # Run recognition only the box hasn't been marked yet
    recognition_cooldowns = {} # {id: last_attempt_time}
    stabilized_boxes = {}

    fps_start_time = time.time() # fps calculation
    frame_count = 0
    display_fps = 0 
    
    # 2. Main loop
    try:
        while True:
            # 2.1, Input: Get the frame
            frame = cam.read()
            
            if frame is None:
                log(f"No frame recieved", "ERROR")
                time.sleep(0.5)
                continue

            # 2.2, Process:

            # ------- FPS Calculation ----------
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                elapsed_seconds = end_time - fps_start_time
                display_fps = frame_count / elapsed_seconds
                
                # Reset window
                fps_start_time = end_time
                frame_count = 0

            # ---------------- Detection & Recognition ---------------------
            detections = detector.detect_and_track(frame)
            current_time = current_time = time.time() # recognition cooldown var

            if detections:
                for target in detections:
                    track_id = target['id']
                    rx1, ry1, rx2, ry2 = target['face_bbox']
                    center = target['center']

                    # ---------------- Box Smoothing ---------------------
                    if track_id not in stabilized_boxes:
                        stabilized_boxes[track_id] = [rx1, ry1, rx2, ry2]
                    
                    # (0.1 = very smooth/slow, 0.9 = jerky/fast)
                    alpha = 0.2 
                    old_box = stabilized_boxes[track_id]
                    
                    # Calculate new smooth coordinates
                    nx1 = int(old_box[0] * (1 - alpha) + rx1 * alpha)
                    ny1 = int(old_box[1] * (1 - alpha) + ry1 * alpha)
                    nx2 = int(old_box[2] * (1 - alpha) + rx2 * alpha)
                    ny2 = int(old_box[3] * (1 - alpha) + ry2 * alpha)
                    
                    stabilized_boxes[track_id] = [nx1, ny1, nx2, ny2]

                    # Check Identity
                    if track_id in identified_targets:
                        target_name = identified_targets[track_id]

                    else:
                        last_try = recognition_cooldowns.get(track_id, 0)
                        
                        if (current_time - last_try) > config.RETRY_INTERVAL:
                            log(f"Attempting recognition for ID: {track_id}...", "INFO")

                            face_crop = frame[ny1:ny2, nx1:nx2]
                            face_crop_fixed = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            
                            target_name, all_scores = recognizer.identify(face_crop_fixed)
                            
                            recognition_cooldowns[track_id] = current_time

                            formatted_scores = json.dumps(all_scores, indent=4)
                            log(f"Distances Analysis:\n{formatted_scores}")
                            
                            if target_name != "Unknown":
                                identified_targets[track_id] = target_name
                                log(f"Target ID {track_id} identified as: {target_name}", "INFO")

                                name = target_name
                            else:
                                log(f"Target ID {track_id} remains Unknown. Cooling down for {config.RETRY_INTERVAL}s", "INFO")
                                name = "Unknown"
                        else:
                            name = "Unknown"

                    # --- VISUALS ---
                    color = (255, 0, 0) if name != "Unknown" else (0, 255, 0)
                    
                    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), color, 2) # Target Box
                    cv2.circle(frame, center, 5, (0, 0, 255), -1) # Center
                    label = f"{name} (ID: {track_id})"
                    cv2.putText(frame, label, (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2) # Label

            # Show FPS on screen
            fps_text = f"FPS: {display_fps:.1f}"
            cv2.putText(frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 2.3, Output: Traking the results for debugging purposes
            if config.DEBUG_MODE:
                cv2.imshow("Sentry Turret V1", frame)

            # 2.4, Exit Condition: Press 'q' to quit, x (1 in this example) ms after the keystroke
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Ctrl+C stops the loop
        log("User interrupted.", "INFO")
        
    finally:
        # 3. Cleanup: Detach hardware
        log("Cleaning up...", "INFO")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()