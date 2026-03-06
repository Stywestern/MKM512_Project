# modules/detector.py

##################################### Imports #####################################
# Libraries
from ultralytics import YOLO
from insightface.model_zoo import get_model

from scipy.spatial import distance as dist

import os
import numpy as np
from abc import ABC, abstractmethod

# Modules
import config
from modules.utils import log

###################################################################################

##################################################################################
#                               Detector Blueprint
##################################################################################

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        pass

##################################################################################
#                               YOLOv8-Lindevs DETECTOR 
##################################################################################

class YOLODetector(BaseDetector):
    def __init__(self, threshold=config.DET_CONF_THRESHOLD):
        """ Get the boxing model from local directory, move it to GPU if we can """

        self.model_name_ = "yolov8n-face-lindevs.pt"
        model_path = os.path.join("assets", "models", self.model_name_)

        self.threshold_ = threshold
        
        try:
            if os.path.exists(model_path):
                log(f"Loading local model from: {model_path}", "INFO")
                self.model_ = YOLO(model_path)
            else:
                log(f"{self.model_name_} not found in assets. Please make sure it is in the right folder", "ERROR")
            
            if config.RUN_ON_GPU:
                self.model_.to('cuda')
                log(f"YOLOv8 successfully loaded on GPU.", "INFO")
            else:
                log("Running on CPU. Performance may be limited.", "WARNING")

        except Exception as e:
            print(f"Failed to initialize detector: {e}", "ERROR")
            raise 

    def __str__(self):
        return f"YOLODetector(Model: {self.model_name_}), Conf_Threshold: %{self.threshold_ * 100}"

    def detect(self, frame):
        """Pure detection: returns [ [x1, y1, x2, y2, conf, cls], ... ]"""
        results = self.model_.predict(source=frame, conf=self.threshold_, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return np.empty((0, 6)) # Return empty array with 6 columns
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Add a dummy class column (0 for face)
        # Result: [x1, y1, x2, y2, confidence, class_id]
        detections = []
        for box, conf in zip(boxes, confs):
            detections.append([*box, conf, 0]) 
            
        return np.array(detections)
    
    
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
    

##################################################################################
#                               SCRFD DETECTOR
##################################################################################

class SCRFDDetector(BaseDetector):
    def __init__(self, threshold=config.DET_CONF_THRESHOLD):
        # SCRFD is usually distributed as an ONNX model
        model_path = os.path.join("assets", "models", "scrfd_10g_bnkps.onnx")
        self.threshold_ = threshold
        
        # Using InsightFace's model zoo for SCRFD as well
        ctx_id = 0 if config.RUN_ON_GPU else -1
        self.model = get_model(model_path, providers=['CUDAExecutionProvider' if config.RUN_ON_GPU else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=ctx_id, input_size=(640, 640))

        log("SCRFD Detector initialized.", "INFO")

    def detect(self, frame):
        """
        Returns: 
        1. boxes: Nx6 numpy array
        2. landmarks: Nx5x2 numpy array
        """
        # SCRFD returns: bboxes [x1, y1, x2, y2, score], kpss [5 landmarks]
        bboxes, kpss = self.model.detect(frame)
        
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 6)), np.empty((0, 5, 2))
        
        # Format for Tracker (BoxMOT needs Nx6)
        detections = np.zeros((bboxes.shape[0], 6))
        detections[:, :5] = bboxes
        
        return detections, kpss