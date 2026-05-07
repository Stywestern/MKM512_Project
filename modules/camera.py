# modules/camera.py

##################################### Imports #####################################
# Libraries
import cv2
import threading
import numpy as np

# Modules
import config
from modules.utils import log

###################################################################################


######################################################################################################################################################################
#                                                                               Classic Camera
######################################################################################################################################################################

class CameraStream:
    """ Handles visual stream from the webcam """

    def __init__(self, src=config.CAMERA_INDEX):
        """ Specs are hardcoded in config, constructor sets and tries the connection """
        self.src_ = src
        self.width_ = config.FRAME_WIDTH
        self.height_ = config.FRAME_HEIGHT

        self.stream_ = cv2.VideoCapture(self.src_, cv2.CAP_MSMF) # init with better usb bus protocol

        self.stream_.set(cv2.CAP_PROP_FRAME_WIDTH, self.width_)
        self.stream_.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height_)
        self.stream_.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        (self.grabbed_, self.frame_) = self.stream_.read()
        
        log("Camera initialized", "INFO")
        
        self.stopped_ = False

    def __str__(self):
        """ Overwrites the print(class) behavior. """
        status = "ACTIVE" if self.stream_.isOpened() else "OFFLINE"
        return f"CameraStream(Index: {self.src_}, Res: {int(self.width_)}x{int(self.height_)}, Status: {status})"

    def start(self):
        """ Starts the async video stream """
        threading.Thread(target=self.update, args=(), daemon=True).start()
        log("Video stream started", "INFO")
        return self

    def update(self):
        """ Pulls the last frame from the feed """
        while True:
            if self.stopped_:
                return
            
            (self.grabbed_, self.frame_) = self.stream_.read()

    def read(self):
        """ To be populated, for now just grabs the frame """
        return self.frame_

    def stop(self):
        """ Kills the async stream, detaching hardware """
        self.stopped = True
        self.stream_.release()


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