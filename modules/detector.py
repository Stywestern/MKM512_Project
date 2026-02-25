# modules/detector.py

##################################### Imports #####################################
# Libraries
from ultralytics import YOLO
from scipy.spatial import distance as dist

from collections import OrderedDict
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
    def detect_and_track(self, frame):
        """Must return a list of dicts that has at least these keys: [{'id': int, 'face_bbox': [x1,y1,x2,y2], 'center': (cx,cy)}]"""
        pass

##################################################################################
#                               YOLOv8-Lindevs DETECTOR
##################################################################################

class YOLODetector(BaseDetector):
    """ The main detection and tracking of faces """

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

    def detect_and_track(self, frame):
        """ Main tracking method, YOLO11 natively has the BoT-SORT algorithm """

        # Run the model to get coordinates
        results = self.model_.track(
            source=frame, 
            persist=True, # Don't lose the object id
            tracker="botsort.yaml", 
            conf=self.threshold_,
            verbose=False
        )

        # Clean data extraction, results[0] contains boxes, names, and IDs for the current frame
        detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Center of the face
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                detections.append({
                    "id": track_id,
                    "face_bbox": [x1, y1, x2, y2],
                    "center": (cx, cy)
                })
        
        return detections
    
##################################################################################
#                               RetinaFace DETECTOR
##################################################################################

# not using this just there for reference purposes
"""
import onnxruntime as ort
ort.set_default_logger_severity(3)

from insightface.model_zoo import get_model

class RetinaDetector(BaseDetector):
    def __init__(self, model_file="assets/models/det_10g.onnx"):
        self.model = get_model(model_file, providers=['CUDAExecutionProvider'])
        # Setting input size: (Width, Height)
        self.model.prepare(ctx_id=0, input_size=(640, 640))

    def detect_and_track(self, frame):
        # returns bboxes and landmarks
        bboxes, kpss = self.model.detect(frame, threshold=0.5)
        
        detections = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            detections.append({
                "id": i, 
                "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": (int((x1+x2)/2), int((y1+y2)/2))
            })
        return detections
"""