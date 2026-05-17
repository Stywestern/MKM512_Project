import cv2
import numpy as np
import time

# Adjust these imports based on your exact folder structure!
# Assuming this script is in the root directory alongside your 'modules' folder.
from modules.detector import SCRFDDetector 
from modules.camera import CameraStream
import config

def estimate_distance_pnp(landmarks, frame_width, frame_height):
    """
    Standalone Perspective-n-Point distance calculator.
    """
    try:
        if landmarks is None or len(landmarks) != 5:
            return None

        # 1. Generic 3D Adult Face Model (in cm)
        model_points = np.array([
            [-3.4, -3.0,  0.0],  # Left Eye
            [ 3.4, -3.0,  0.0],  # Right Eye
            [ 0.0,  0.0, -3.0],  # Nose Tip (Sticking out)
            [-2.6,  4.0,  0.0],  # Left Mouth
            [ 2.6,  4.0,  0.0]   # Right Mouth
        ], dtype=np.float32)

        # 2. 2D Image Points from SCRFD
        image_points = np.array(landmarks, dtype=np.float32)

        # 3. Camera Intrinsics
        focal_length = config.FOCAL_LENGTH
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        # 4. Solve PnP
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
    print("[INFO] Booting Sandbox Environment...")
    
    # 1. Initialize the Threaded CameraStream
    print("[INFO] Initializing Async Camera...")
    cam = CameraStream().start()
    
    # Wait a split second to ensure the thread has pulled the first frame
    time.sleep(0.5)
    
    actual_w = cam.width_
    actual_h = cam.height_
    print(f"[INFO] Camera running at: {actual_w}x{actual_h}")

    # 2. Initialize only the Detector
    print("[INFO] Loading SCRFD Model...")
    detector = SCRFDDetector()
    print("[INFO] Ready. Press 'q' to quit.")

    while True:
        start_time = time.time()
        
        # 3. Pull from the Async Stream
        raw_frame = cam.read()
        
        if raw_frame is None:
            time.sleep(0.01) # Micro-sleep to prevent CPU thrashing if frame isn't ready
            continue

        # CRITICAL: Memory Isolation! 
        # Copy the frame so we don't draw directly onto the background thread's buffer
        frame = raw_frame.copy()

        # Run detection
        boxes, landmarks_list, _ = detector.detect(frame)

        # Process each face found
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i].astype(int)
                landmarks = landmarks_list[i].astype(int) if (landmarks_list is not None and len(landmarks_list) > i) else None
                
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if landmarks is not None and len(landmarks) == 5:
                    # Draw the 5 facial landmarks to verify SCRFD tracking
                    colors = [(0,255,0), (0,255,0), (0,0,255), (255,0,0), (255,0,0)] # Eyes, Nose, Mouth
                    for (lx, ly), color in zip(landmarks, colors):
                        cv2.circle(frame, (lx, ly), 3, color, -1)

                    # --- RUN THE PNP TEST ---
                    distance_cm = estimate_distance_pnp(landmarks, actual_w, actual_h)

                    if distance_cm:
                        # Display the calculated distance
                        text = f"Dist: {distance_cm:.1f} cm"
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Dist: CALC FAILED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS calculation
        delta_time = time.time() - start_time
        fps = 1.0 / max(delta_time, 0.001) # Caps maximum measurable FPS at 1000
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show the sandbox feed
        cv2.imshow("PnP Distance Sandbox", frame)

        # Press 'q' to exit safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Safely close the background thread and window
    print("[INFO] Shutting down...")
    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()