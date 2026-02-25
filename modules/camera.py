# modules/camera.py

##################################### Imports #####################################
# Libraries
import cv2
import threading

# Modules
import config
from modules.utils import log

###################################################################################

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