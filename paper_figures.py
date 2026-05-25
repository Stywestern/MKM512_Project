try:
    import onnxruntime as ort
    # This pre-warms the DLL bindings before PyQt6 can interfere
    _ = ort.get_device() 

except Exception as e:
    print(f"Pre-import warning: {e}")

import cv2
import time
import os
import numpy as np

# Adjust imports to match your VSCode workspace
import config
from modules.camera import CameraStream
from modules.detector import SCRFDDetector
from modules.tracker import BoTSORTTracker
from modules.recognizer import TurretRecognizer

def estimate_distance_pnp(landmarks):
    """
    Perspective-n-Point distance calculator.
    Uses your specific A4Tech focal length and nose-origin 3D model.
    """
    try:
        if landmarks is None or len(landmarks) != 5:
            return None

        # Generic 3D Adult Face Model (Origin is Nose Tip)
        model_points = np.array([
            [-3.4, -3.0,  3.0],  # Left Eye 
            [ 3.4, -3.0,  3.0],  # Right Eye
            [ 0.0,  0.0,  0.0],  # Nose Tip 
            [-2.6,  4.0,  3.0],  # Left Mouth
            [ 2.6,  4.0,  3.0]   # Right Mouth
        ], dtype=np.float32)

        image_points = np.array(landmarks, dtype=np.float32)

        # A4Tech PK-910H Intrinsics (Fallback to 1573.0 if missing from config)
        focal_length = getattr(config, 'FOCAL_LENGTH', 1573.0)
        center_x = config.FRAME_WIDTH / 2.0
        center_y = config.FRAME_HEIGHT / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP using SQPNP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )

        if success:
            z_distance_cm = float(translation_vec[2][0]) 
            if 10.0 < z_distance_cm < 1000.0:
                return z_distance_cm

        return None
        
    except Exception as e:
        print(f"PnP Math Error: {e}")
        return None

def main():
    print("[INFO] Booting Complete AI Pipeline...")
    
    cam = CameraStream().start()
    time.sleep(0.5) 
    
    detector = SCRFDDetector()
    tracker = BoTSORTTracker()
    recognizer = TurretRecognizer()

    print("[INFO] Pipeline ready.")
    print("[INFO] Press 's' to save a snapshot with aligned crops.")
    print("[INFO] Press 'q' to quit.")

    os.makedirs("paper_assets", exist_ok=True)
    snapshot_counter = 1
    
    # Target Memory Dictionary (Mirrors VisionWorker)
    active_targets = {}

    # Safety fallbacks for config colors
    color_enemy = getattr(config, 'COLOR_ENEMY', (0, 0, 255))
    color_friend = getattr(config, 'COLOR_FRIEND', (0, 255, 0))
    color_stranger = getattr(config, 'COLOR_STRANGER', (255, 255, 0))

    while True:
        raw_frame = cam.read()
        if raw_frame is None:
            time.sleep(0.01)
            continue

        frame = raw_frame.copy()
        ai_frame = raw_frame.copy()
        pristine_frame = raw_frame.copy()

        # Step 1: Detect and Track
        raw_boxes, landmarks, _ = detector.detect(ai_frame)
        detections = tracker.update(raw_boxes, ai_frame)

        # Cache valid landmarks for the snapshot trigger
        snapshot_cache = []

        for target in detections:
            track_id = target["id"]
            cx, cy = target["center"]
            sx1, sy1, sx2, sy2 = [int(v) for v in target["face_bbox"]]

            if landmarks is not None and len(landmarks) > 0:
                # Spatial Sync
                lm_idx = np.argmin([
                    np.linalg.norm(np.array([cx, cy]) - np.mean(lm, axis=0)) 
                    for lm in landmarks
                ])
                target_landmarks = landmarks[lm_idx]
                
                # Calculate True 3D Distance
                current_dist = estimate_distance_pnp(target_landmarks)

                # --- RECOGNITION MEMORY LOGIC ---
                current_time = time.time()
                needs_identification = True
                
                if track_id in active_targets:
                    if active_targets[track_id]["name"] != "Unknown":
                        needs_identification = False
                    elif (current_time - active_targets[track_id]["last_auth"]) < 5.0:
                        needs_identification = False

                if needs_identification:
                    # Run full heavy identification
                    name, _, _ = recognizer.identify(pristine_frame, target_landmarks)
                    active_targets[track_id] = {
                        "name": name,
                        "last_auth": current_time,
                        "distance": current_dist or 200.0
                    }
                else:
                    # Update distance only
                    if current_dist is not None:
                        active_targets[track_id]["distance"] = current_dist
                    name = active_targets[track_id]["name"]

                # Cache this target's pristine data in case we press 's'
                snapshot_cache.append((track_id, target_landmarks))
                
                # --- AFFILIATION & COLOR MAPPING ---
                if name in getattr(config, 'ENEMIES', []):
                    affiliation = "ENEMY"
                    color = color_enemy
                elif name in getattr(config, 'FRIENDS', []):
                    affiliation = "FRIEND"
                    color = color_friend
                else:
                    affiliation = "STRANGER"
                    color = color_stranger

                dist_display = active_targets[track_id].get("distance", 200.0)

                # --- DRAW HUD ---
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.rectangle(frame, (sx1, sy1 - 22), (sx2, sy1), color, -1)

                display_text = f"{affiliation}: {name} (ID:{track_id})(DIST: {dist_display:.1f}cm)"
                cv2.putText(frame, display_text, (sx1 + 5, sy1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw the 5 SCRFD landmarks
                lm_colors = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 0)]
                for (lx, ly), lc in zip(target_landmarks, lm_colors):
                    cv2.circle(frame, (int(lx), int(ly)), 4, lc, -1)

        cv2.imshow("Academic Figure Generator", frame)

        # --- CAPTURE LOGIC ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            main_filename = f"paper_assets/hybrid_pipeline_fig_{snapshot_counter}_main.jpg"
            cv2.imwrite(main_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"\n[SUCCESS] Main scene saved to {main_filename}")
            
            # Instantly generate crops for everyone in the cached frame
            for tid, t_landmarks in snapshot_cache:
                _, _, aligned_face = recognizer.identify(pristine_frame, t_landmarks)
                if aligned_face is not None and aligned_face.size > 0:
                    align_filename = f"paper_assets/hybrid_pipeline_fig_{snapshot_counter}_aligned_id_{tid}.jpg"
                    cv2.imwrite(align_filename, aligned_face, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    print(f"[SUCCESS] Aligned crop for ID {tid} saved to {align_filename}")
            
            snapshot_counter += 1
            
            flash_frame = np.ones_like(frame) * 255
            cv2.imshow("Academic Figure Generator", flash_frame)
            cv2.waitKey(50)
            
        elif key == ord('q'):
            break

    print("[INFO] Shutting down...")
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()