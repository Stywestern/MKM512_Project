# face_embeddings.py

import os
import pickle
import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align

# --- CONFIGURATION ---
RAW_IMAGES_PATH = "assets/faces/raw_images"
# Using the specific model filenames from buffalo_l
DET_MODEL_PATH = "assets/models/det_10g.onnx"
REC_MODEL_PATH = "assets/models/w600k_r50.onnx"
EMBEDDINGS_FILE = os.path.join("assets/faces/embeddings", "w600k_r50_encodings.pkl")

def update_embeddings():
    # 1. Load Decoupled Models
    det_model = get_model(DET_MODEL_PATH, providers=['CUDAExecutionProvider'])
    det_model.prepare(ctx_id=0, input_size=(640, 640))
    
    rec_model = get_model(REC_MODEL_PATH, providers=['CUDAExecutionProvider'])
    rec_model.prepare(ctx_id=0)

    known_data = []
    
    for person_name in os.listdir(RAW_IMAGES_PATH):
        person_dir = os.path.join(RAW_IMAGES_PATH, person_name)

        if not os.path.isdir(person_dir): continue
        #if person_dir != "assets/faces/raw_images\Kerem_Cantimur": continue

        for image_name in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, image_name))
            #cv2.imshow("preview", img)
            #cv2.waitKey(0)

            # Checkpoint 0: Raw Image Verification
            if img is None: continue
            
            # --- ATTEMPT 1: Original Image ---
            bboxes, kpss = det_model.detect(img, max_num=1)
            source_to_use = img
            
            # --- ATTEMPT 2: Conditional Padding (The Fallback) ---
            if bboxes.shape[0] == 0:
                print(f"[-] No face in original {image_name}. Attempting padded fallback...")
                h, w = img.shape[:2]
                pad = max(h, w) // 2
                source_to_use = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                bboxes, kpss = det_model.detect(source_to_use, max_num=1)

            # Final Check
            if bboxes.shape[0] == 0:
                print(f"[!] Critical: Could not find face in {image_name} even with padding. Skipping.")
                continue
            
            #cv2.imshow("padded", source_to_use)
            #cv2.waitKey(0)

            bboxes, kpss = det_model.detect(source_to_use, max_num=1)
            
            if bboxes.shape[0] == 0:
                print(f"Warning: No face found in {image_name}")
                continue

            # Checkpoint 1: Visualizing Bbox and Landmarks (kps)
            debug_img = source_to_use.copy()
            x1, y1, x2, y2, score = bboxes[0]
            cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            for point in kpss[0]: # Draw the 5 landmarks
                cv2.circle(debug_img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
            
            #cv2.imshow("Snapshot: Detection & Landmarks", debug_img)
            #cv2.waitKey(0)

            # Step B: Create the "Face" object
            from insightface.app.common import Face
            face = Face(bbox=bboxes[0][:4], kps=kpss[0], det_score=bboxes[0][4])
            
            # Checkpoint 2: Inspecting the Face object structure
            print(f"\n--- Face Object Snapshot for {image_name} ---")
            print(f"Bbox: {face.bbox}")
            print(f"Landmarks (kps):\n{face.kps}")
            print(f"Detection Score: {face.det_score:.4f}")

            # Step C: Get Embedding
            embedding = rec_model.get(source_to_use, face) 
            
            # Checkpoint 3: Raw Embedding Preview (First 5 values)
            print(f"Raw Embedding (shape {embedding.shape}): {embedding[:5]}...")

            # Step D: L2 Normalization
            norm = np.linalg.norm(embedding)
            normed_embedding = embedding / norm
            
            # Checkpoint 4: Normed Embedding Verification
            # A properly normed vector should have a magnitude of 1.0
            print(f"Normed Embedding (shape {normed_embedding.shape}): {normed_embedding[:5]}...")
            print(f"Vector Magnitude (Should be 1.0): {np.linalg.norm(normed_embedding):.4f}")
            print("-------------------------------------------\n")

            known_data.append({
                "name": person_name,
                "embedding": normed_embedding,
                "origin": image_name
            })

            print(f"Encoded: {person_name} - {image_name}")

    # 3. Save
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known_data, f)
    print(f"Database Rebuilt. Total: {len(known_data)}")

if __name__ == "__main__":
    update_embeddings()