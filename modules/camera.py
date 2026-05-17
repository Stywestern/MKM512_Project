# modules/camera.py

##################################### Imports #####################################
# Libraries
import cv2
import threading
import numpy as np
import time

# Modules
import config
from modules.utils import log

###################################################################################

######################################################################################################################################################################
#                                                                               Classic Camera
######################################################################################################################################################################

class CameraStream:
    """ Handles thread-safe visual stream from the webcam """

    def __init__(self, src=config.CAMERA_INDEX):
        self.src_ = src
        self.width_ = config.FRAME_WIDTH
        self.height_ = config.FRAME_HEIGHT

        # 1. Threading Lock to prevent memory collisions
        self.lock = threading.Lock()
        
        self.stream_ = cv2.VideoCapture(self.src_, cv2.CAP_MSMF)
        self.stream_.set(cv2.CAP_PROP_FRAME_WIDTH, self.width_)
        self.stream_.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height_)
        self.stream_.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 2. Hardware Sanity Check
        if not self.stream_.isOpened():
            log(f"CRITICAL: Camera at index {self.src_} failed to open!", "ERROR")
            self.grabbed_ = False
            self.frame_ = None
        else:
            (self.grabbed_, self.frame_) = self.stream_.read()
            log(f"Camera initialized (Index {self.src_})", "INFO")
        
        self.stopped_ = False

    def __str__(self):
        status = "ACTIVE" if self.stream_.isOpened() else "OFFLINE"
        return f"CameraStream(Index: {self.src_}, Res: {int(self.width_)}x{int(self.height_)}, Status: {status})"

    def start(self):
        """ Starts the async video stream """
        if self.stream_.isOpened():
            threading.Thread(target=self.update, args=(), daemon=True).start()
            log("Video stream thread started", "INFO")
        return self

    def update(self):
        """ Pulls the last frame from the feed safely """
        while not self.stopped_:
            grabbed, frame = self.stream_.read()
            
            # 3. Hardware Fault Tolerance: If the camera stutters, don't crash
            if not grabbed or frame is None:
                time.sleep(0.01) # Wait 10ms for USB bus to recover
                continue
                
            # Safely lock the memory, update the frame, and release the lock
            with self.lock:
                self.grabbed_ = grabbed
                self.frame_ = frame

    def read(self):
        """ Returns a safe, locked copy of the current frame """
        with self.lock:
            # We return the reference safely. The VisionWorker takes care of .copy()
            return self.frame_

    def stop(self):
        """ Kills the async stream, detaching hardware """
        # THE FIX: Corrected the typo to self.stopped_
        self.stopped_ = True 
        
        if self.stream_.isOpened():
            self.stream_.release()
        log("Camera hardware released.", "WARNING")


######################################################################################################################################################################
#                                                                      Rotated Camera
######################################################################################################################################################################

class RotatedCameraStream(CameraStream):
    """ 
    Specialized stream for rotated hardware mounts.
    Corrects orientation at the source to keep the AI pipeline upright.
    """
    def __init__(self, src=config.CAMERA_INDEX, rotation=cv2.ROTATE_90_CLOCKWISE):
        # Initialize the base class first
        super().__init__(src)
        self.rotation_type = rotation
        self.frame_ = np.zeros((self.height_, self.width_, 3), dtype=np.uint8)
        
        # Correct the dimensions once for external callers
        if self.grabbed_:
            temp_frame = cv2.rotate(self.frame_, self.rotation_type)
            self.height_, self.width_ = temp_frame.shape[:2]
            log(f"RotatedCamera initialized. New Virtual Res: {self.width_}x{self.height_}", "INFO")

    def update(self):
        """ Overrides the base update to inject the rotation logic """
        while True:
            if self.stopped_:
                return
            
            (grabbed, raw_frame) = self.stream_.read()
            if grabbed:
                # Rotates the frame before saving it to the buffer
                self.frame_ = cv2.rotate(raw_frame, self.rotation_type)
                self.grabbed_ = grabbed