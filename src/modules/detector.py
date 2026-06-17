# src/modules/detector.py

##################################### Imports #####################################
# Standart Libraries
import os
import numpy as np
from abc import ABC, abstractmethod

# Third Party Libraries
from insightface.model_zoo import get_model

# Modules
import src.config as config
from src.modules.utils import log

###################################################################################

##################################################################################
#                               Detector Blueprint
##################################################################################

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        pass


##################################################################################
#                               SCRFD DETECTOR
##################################################################################

class SCRFDDetector(BaseDetector):
    def __init__(self, threshold=config.DET_CONF_THRESHOLD):
        # SCRFD is usually distributed as an ONNX model
        self.model_path_ = os.path.join("assets", "models", "scrfd_10g_bnkps.onnx")
        self.threshold_ = threshold

        self.focal_length = config.FOCAL_LENGTH # calibration
        self.real_ipd = 6.3 # Average human eye distance in cm
        
        # Using InsightFace's model zoo for SCRFD
        ctx_id = 0 if config.RUN_ON_GPU else -1
        self.model = get_model(self.model_path_, providers=['CUDAExecutionProvider' if config.RUN_ON_GPU else 'CPUExecutionProvider'])
        self.model.prepare(
                ctx_id=ctx_id,
                input_size=(640, 640),
                det_thresh=self.threshold_
            )

        log("SCRFD Detector initialized.", "INFO")

    def __str__(self):
        return f"SCRFD Detector (Model: {self.model_path}), Conf_Threshold: %{self.threshold_ * 100}"

    def detect(self, frame):
        """
        Returns: 
        1. boxes: Nx6 numpy array
        2. landmarks: Nx5x2 numpy array
        """
        # SCRFD returns: bboxes [x1, y1, x2, y2, score], kpss [5 landmarks]
        bboxes, kpss = self.model.detect(frame)
        
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 6)), np.empty((0, 5, 2)), []
        
        # Format for Tracker (BoxMOT needs Nx6)
        detections = np.zeros((bboxes.shape[0], 6))
        detections[:, :5] = bboxes
        
        return detections, kpss

##################################################################################
#                               RetinaFace DETECTOR
##################################################################################

class RetinaDetector(BaseDetector):
    def __init__(self, threshold=config.DET_CONF_THRESHOLD):
        # Path to the ONNX model (usually det_10g.onnx or det_500m.onnx)
        model_path = os.path.join("assets", "models", "det_10g.onnx")
        self.threshold_ = threshold
        
        # ctx_id=0 uses the first GPU, -1 for CPU
        ctx_id = 0 if config.RUN_ON_GPU else -1
        self.model = get_model(model_path, providers=['CUDAExecutionProvider' if config.RUN_ON_GPU else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=ctx_id, input_size=(640, 640))

        log("RetinaFace Detector initialized.", "INFO")

    def detect(self, frame):
        # bboxes: [x1, y1, x2, y2, score]
        bboxes, kpss = self.model.detect(frame)
        
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 6))
        
        # RetinaFace returns [x1, y1, x2, y2, score]. 
        # We append a 0 for the 'class' column.
        detections = []
        for box in bboxes:
            detections.append([box[0], box[1], box[2], box[3], box[4], 0])
            
        return np.array(detections)
    

