# modules/recognizer.py

##################################### Imports #####################################
# Libraries
import os
import pickle
import numpy as np
import cv2

import onnxruntime as ort

from insightface.model_zoo import get_model
from insightface.utils import face_align

# Modules
import config
from modules.utils import log

###################################################################################

class TurretRecognizer:
    def __init__(self, model_name="w600k_r50", threshold=config.REG_CONF_THRESHOLD):
        self.model_name_ = model_name
        self.threshold_ = threshold
        
        rec_file = os.path.join("assets", "models", "w600k_r50.onnx")
        
        try:
            # Load ArcFace only
            from insightface.model_zoo import get_model
            self.rec_model = get_model(rec_file, providers=['CUDAExecutionProvider'])
            self.rec_model.prepare(ctx_id=0)
            
            log(f"Recognition model loaded: {rec_file}", "INFO")

        except Exception as e:
            log(f"Failed to load ArcFace: {e}", "ERROR")
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

    def identify(self, full_frame, landmarks):
        empty_img = np.array([], dtype=np.uint8) # no image placeholder

        # 1. Alignment
        try:
            aligned_face = face_align.norm_crop(full_frame, landmark=landmarks)
        except Exception as e:
            log(f"Alignment failed: {e}", "WARNING")
            return "Unknown", {}, empty_img

        # 2. ArcFace Feature Extraction
        raw_embedding = self.rec_model.get_feat(aligned_face)
        
        # 3. Normalization (Unit Vector for Cosine Similarity)
        norm = np.linalg.norm(raw_embedding)
        current_embedding = raw_embedding / norm

        # 4. Database Comparison (Cosine Similarity)
        best_name = "Unknown"
        min_dist = 1.0
        debug_distances = {}

        for entry in self.db_:
            # Dot product of normalized vectors = Cosine Similarity
            similarity = np.dot(current_embedding, entry["embedding"])
            dist = float(1.0 - similarity.item())
            
            origin_key = entry["origin"] 
            debug_distances[origin_key] = round(dist, 4)
            
            if dist < min_dist:
                min_dist = dist
                best_name = entry["name"]

        # 5. Threshold Verification
        if min_dist > self.threshold_:
            return "Unknown", debug_distances, aligned_face
        
        return best_name, debug_distances, aligned_face