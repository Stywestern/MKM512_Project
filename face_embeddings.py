# face_embeddings.py
import os, pickle, cv2, numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align
from modules.utils import log

# --- CONFIG ---
RAW_IMAGES_PATH = "assets/faces/raw_images"
SCRFD_MODEL_PATH = "assets/models/scrfd_10g_bnkps.onnx"
REC_MODEL_PATH = "assets/models/w600k_r50.onnx"
EMBEDDINGS_FILE = "assets/faces/embeddings/w600k_r50_encodings.pkl"
DEBUG_PATH = "assets/faces/debug_aligned" # New debug directory

def update_embeddings():
    det_model = get_model(SCRFD_MODEL_PATH, providers=['CUDAExecutionProvider'])
    det_model.prepare(ctx_id=0, input_size=(640, 640))
    rec_model = get_model(REC_MODEL_PATH, providers=['CUDAExecutionProvider'])
    rec_model.prepare(ctx_id=0)

    known_data = []
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            known_data = pickle.load(f)
    
    existing_origins = {entry["origin"] for entry in known_data}
    new_entries = 0

    for person_name in os.listdir(RAW_IMAGES_PATH):
        person_dir = os.path.join(RAW_IMAGES_PATH, person_name)
        if not os.path.isdir(person_dir): continue

        # Prepare debug subdir for this person
        person_debug_dir = os.path.join(DEBUG_PATH, person_name)
        os.makedirs(person_debug_dir, exist_ok=True)

        for image_name in os.listdir(person_dir):
            if image_name in existing_origins: continue

            img = cv2.imread(os.path.join(person_dir, image_name))
            if img is None: continue

            # --- ATTEMPT 1: Direct Detection ---
            bboxes, kpss = det_model.detect(img)
            source_to_use = img

            # --- ATTEMPT 2: The Padding Fallback ---
            if bboxes is None or bboxes.shape[0] == 0:
                h, w = img.shape[:2]
                pad = max(h, w) // 2
                source_to_use = cv2.copyMakeBorder(img, pad, pad, pad, pad, 
                                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
                bboxes, kpss = det_model.detect(source_to_use)

            if bboxes is not None and bboxes.shape[0] > 0:
                # 1. Align the face
                aligned_face = face_align.norm_crop(source_to_use, landmark=kpss[0])
                
                # 2. SAVE FOR DEBUGGING
                # This lets you see exactly what is being sent to the recognizer
                debug_file_path = os.path.join(person_debug_dir, f"aligned_{image_name}")
                cv2.imwrite(debug_file_path, aligned_face)

                # 3. Generate Embedding
                embedding = rec_model.get_feat(aligned_face)
                normed_embedding = embedding / np.linalg.norm(embedding)

                known_data.append({
                    "name": person_name,
                    "embedding": normed_embedding,
                    "origin": image_name
                })
                new_entries += 1
                print(f"[+] Encoded & Saved Debug: {person_name} ({image_name})")
            else:
                print(f"[!] FAILED: {image_name} - No face found even with padding.")

    if new_entries > 0:
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_data, f)
        print(f"Success. Total Database size: {len(known_data)}")

if __name__ == "__main__":
    update_embeddings()