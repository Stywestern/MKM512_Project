import cv2
import threading
import time
import math

class CameraTester:
    """ Standalone test class stripped of external dependencies. """
    def __init__(self, src=0, req_width=1920, req_height=1080):
        self.src = src
        self.req_w = req_width
        self.req_h = req_height

        # Using CAP_DSHOW for 1080p on Windows. It's often much faster for high-res webcams than MSMF.
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)

        # Force the resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Read the first frame to ensure connection
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
        # Verify what resolution the camera ACTUALLY accepted
        self.actual_w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate Aspect Ratio
        gcd = math.gcd(self.actual_w, self.actual_h)
        self.aspect_w = self.actual_w // gcd
        self.aspect_h = self.actual_h // gcd
        
        print(f"[INFO] Camera {self.src} initialized.")
        print(f"[INFO] Requested: {self.req_w}x{self.req_h}")
        print(f"[INFO] Actual output: {self.actual_w}x{self.actual_h} ({self.aspect_w}:{self.aspect_h})")

    def start(self):
        """ Starts the async video stream """
        threading.Thread(target=self.update, daemon=True).start()
        print("[INFO] Video stream started. Press 'q' to quit.")
        return self

    def update(self):
        """ Pulls the last frame from the feed continuously """
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """ Returns the current frame """
        return self.frame

    def stop(self):
        """ Kills the async stream, detaching hardware """
        self.stopped = True
        self.stream.release()

if __name__ == "__main__":
    # Initialize requesting full 1080p. 
    cam = CameraTester(src=1, req_width=1920, req_height=1080).start()
    
    prev_time = time.time()
    
    while True:
        frame = cam.read()
        
        if frame is None or frame.size == 0:
            time.sleep(0.01)
            continue
            
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # --- HUD OVERLAY ---
        # Draw a center crosshair to check alignment
        cx, cy = cam.actual_w // 2, cam.actual_h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
        
        # Telemetry Data
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"RES: {cam.actual_w}x{cam.actual_h}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"RATIO: {cam.aspect_w}:{cam.aspect_h}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        cv2.imshow("Hardware Test - A4Tech", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Shutting down...")
            break
            
    cam.stop()
    cv2.destroyAllWindows()