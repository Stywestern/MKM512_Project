import cv2
import time

def test_camera(index=1):
    # Try with DirectShow (DSHOW) for better Windows compatibility
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {index}")
        return

    print(f"Camera {index} connected. Press 'q' to exit.")
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Calculate live FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display info
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sentry Debug - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If 0 doesn't work, try 1 or 2
    test_camera(0)