# src/modules/camera.py

##################################### Imports #####################################
# Standart Libraries
import threading
import numpy as np
import time

# Third Party Libraries
import cv2

# Modules
import src.config as config
from src.modules.utils import log

###################################################################################

######################################################################################################################################################################
#                                                                               Classic Camera
######################################################################################################################################################################

class CameraStream:
    """ Handles thread-safe, fault-tolerant visual stream from the webcam """

    def __init__(self, src=config.CAMERA_INDEX):
        self.src_ = src
        self.width_ = config.FRAME_WIDTH
        self.height_ = config.FRAME_HEIGHT

        # Threading Lock to prevent memory collisions
        self.lock = threading.Lock()
        
        self.stream_ = cv2.VideoCapture(self.src_, cv2.CAP_MSMF)
        self.stream_.set(cv2.CAP_PROP_FRAME_WIDTH, self.width_)
        self.stream_.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height_)
        self.stream_.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Hardware Sanity Check
        if not self.stream_.isOpened():
            log(f"CRITICAL: Camera at index {self.src_} failed to open", "ERROR")
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
        threading.Thread(target=self.update, args=(), name="CameraThread", daemon=True).start()
        log("Video stream thread started", "INFO")
        return self

    def _reconnect_hardware(self):
        """ The Polling Loop: Tries to re-establish connection to the USB hardware """
        log("CAMERA WATCHDOG: Entering recovery mode...", "WARNING")
        
        while not self.stopped_ and not self.stream_.isOpened():
            log(f"CAMERA WATCHDOG: Polling hardware node {self.src_}...", "DEBUG")
            time.sleep(1.0)  # Wait 1 second between pings to avoid locking the OS USB bus
            
            self.stream_ = cv2.VideoCapture(self.src_, cv2.CAP_MSMF)
            
            if self.stream_.isOpened():
                # The hardware came back, re-apply all configurations
                self.stream_.set(cv2.CAP_PROP_FRAME_WIDTH, self.width_)
                self.stream_.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height_)
                self.stream_.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                log("CAMERA WATCHDOG: Hardware recovered successfully!", "SUCCESS")
                break

    def update(self):
        """ Pulls the last frame from the feed """
        fail_count = 0
        max_fails = 15  # ~150-300ms of dead air means the cable was unplugged
        
        # --- THREAD DIAGNOSTIC: Heartbeat Timer ---
        last_heartbeat = time.time()

        while not self.stopped_:
            # --- THREAD DIAGNOSTIC: 10-Second Ping ---
            current_time = time.time()
            if current_time - last_heartbeat >= 10.0:
                log("HEARTBEAT: Camera loop is alive, active, and pulling frames.", "DEBUG")
                last_heartbeat = current_time

            if not self.stream_.isOpened():
                self._reconnect_hardware()
                continue

            grabbed, frame = self.stream_.read()
            
            # Hardware Fault Tolerance / Drop Detection
            if not grabbed or frame is None:
                fail_count += 1
                if fail_count > max_fails:
                    log("CAMERA WATCHDOG: Connection lost! Releasing dead hardware...", "ERROR")
                    self.stream_.release()  # Force kill the ghost pointer
                    
                    with self.lock:
                        self.frame_ = None  # Safely blanks out the pipeline
                
                time.sleep(0.02)  # Wait 20ms before trying to read again
                continue
                
            # If the frame was successful, reset counter
            fail_count = 0
            
            # Lock the memory, update the frame, and release the lock
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
        self.stopped_ = True 
        
        if self.stream_.isOpened():
            self.stream_.release()
        log("Camera hardware released.", "WARNING")
