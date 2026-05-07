import cv2
import numpy as np
from modules.detector import SCRFDDetector
from modules.camera import CameraStream

def run_calibration():
    detector = SCRFDDetector(threshold=0.5)
    cam = CameraStream(0)
    
    print("--- SENTRY CALIBRATION MODE ---")
    print("1. Stand exactly 200cm away from the lens.")
    print("2. Watch the terminal for 'Current Pixel IPD'.")
    print("3. Press ESC to stop and calculate the Focal Length.")

    pixel_distances = []

    while True:
        frame = cam.read()
        if frame is None: break

        # Detect
        _, landmarks, _ = detector.detect(frame)

        if len(landmarks) > 0:
            face_lms = landmarks[0]
            dist = np.linalg.norm(face_lms[0] - face_lms[1])
            pixel_distances.append(dist)
            
            # This will let you see the value even if the window is acting up
            print(f"Current Pixel IPD: {dist:.2f} | Samples: {len(pixel_distances)}", end='\r')

            # Visual Feedback
            cv2.line(frame, tuple(face_lms[0].astype(int)), tuple(face_lms[1].astype(int)), (0, 255, 0), 2)
            cv2.putText(frame, f"Pixels: {dist:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        
        # Check for ESC (27) or 'q' (113)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    if pixel_distances:
        # Use a mean of the samples for higher precision
        avg_pixels = sum(pixel_distances) / len(pixel_distances)
        focal_length = (150 * avg_pixels) / 6.3
        
        print(f"\n\nFinal Average Pixels: {avg_pixels:.2f}")
        print(f"SET YOUR FOCAL_LENGTH TO: {focal_length:.2f}")
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration()