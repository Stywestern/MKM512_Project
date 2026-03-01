# modules/recognizer.py

##################################### Imports #####################################
# Libraries
import os
import pickle
import numpy as np
import cv2

import onnxruntime as ort
ort.set_default_logger_severity(3)

from insightface.model_zoo import get_model
from insightface.app.common import Face
from insightface.utils import face_align

# Modules
import config
from modules.utils import log

###################################################################################

class TurretRecognizer:
    def __init__(self, model_name="w600k_r50", threshold=config.REG_CONF_THRESHOLD):
        self.model_name_ = model_name
        self.threshold_ = threshold
        
        det_file = os.path.join("assets", "models", "det_10g.onnx")
        rec_file = os.path.join("assets", "models", "w600k_r50.onnx")
        
        try:
            # Load RetinaFace 
            self.det_model = get_model(det_file, providers=['CUDAExecutionProvider'])
            self.det_model.prepare(ctx_id=0, input_size=(640, 640))
            
            # Load ArcFace 
            self.rec_model = get_model(rec_file, providers=['CUDAExecutionProvider'])
            self.rec_model.prepare(ctx_id=0)
            
            log(f"Recognition models loaded: {det_file} & {rec_file}", "INFO")
        except Exception as e:
            log(f"Failed to load models: {e}", "ERROR")
            raise

        self.db_ = []
        self.load_database(model_name)

    def load_database(self, model_name):
        db_path = os.path.join("assets/faces/embeddings", f"{model_name}_encodings.pkl")
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.db_ = pickle.load(f)
            log(f"Loaded {len(self.db_)} embeddings for {model_name}", "INFO")
        else:
            log(f"No database found at {db_path}", "WARNING")

    def identify(self, face_crop):
        empty_img = np.array([], dtype=np.uint8) # no image placeholder

        # EXIT 1: Invalid input
        if face_crop is None or face_crop.size == 0:
            return "Unknown", {}, empty_img

        # 2. Adaptive Landmark Detection (RetinaFace)
        bboxes, kpss = self.det_model.detect(face_crop, max_num=1)
        source_to_use = face_crop
        
        if bboxes.shape[0] == 0:
            h, w = face_crop.shape[:2]
            pad = max(h, w) // 2
            source_to_use = cv2.copyMakeBorder(face_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            bboxes, kpss = self.det_model.detect(source_to_use, max_num=1)

        # EXIT 2: No face found even after padding
        if bboxes.shape[0] == 0:
            return "Unknown", {}, empty_img

        # 3. Create Face object & Internal Alignment
        aligned_face = face_align.norm_crop(source_to_use, landmark=kpss[0]) # This is the 112x112 'straightened' face for the [ALIGN] box
        
        # ArcFace features
        raw_embedding = self.rec_model.get_feat(aligned_face) 
        
        # 4. Normalization
        norm = np.linalg.norm(raw_embedding)
        current_embedding = raw_embedding / norm

        # 5. Database Comparison
        best_name = "Unknown"
        min_dist = 1.0  # Maximum possible distance for Cosine
        debug_distances = {}

        for entry in self.db_:
            similarity = np.dot(current_embedding, entry["embedding"]) # Standard Dot Product (Similarity: 1.0 is perfect)
            dist = float(1.0 - similarity.item()) # Convert to Distance (0.0 is perfect)
            
            origin_key = entry["origin"] 
            debug_distances[origin_key] = round(dist, 4)
            
            if dist < min_dist:
                min_dist = dist
                best_name = entry["name"]

        if min_dist > self.threshold_:
            return "Unknown", debug_distances, aligned_face
        
        log(f"Recognized: {best_name} (Dist: {min_dist:.4f})", "SUCCESS")
        return best_name, debug_distances, aligned_face