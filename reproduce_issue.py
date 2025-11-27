import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(os.getcwd())

# Import app to trigger initialization (ensure_classifier_ready)
from app import app, ensure_classifier_ready, match_persisted_embedding, is_known
import model_utils

def reproduce():
    print("Starting reproduction script...")
    
    # Ensure classifier is ready (simulating app startup)
    ensure_classifier_ready()
    
    # Check classifier status
    clf = model_utils.load_classifier()
    if clf:
        print(f"Classifier loaded: {clf.get('method')}")
    else:
        print("Classifier STILL NOT LOADED after ensure_classifier_ready")
        
    # Check embeddings
    from app import _load_persisted_embeddings
    embs, labels = _load_persisted_embeddings()
    if embs is not None:
        print(f"Persisted embeddings: {len(labels)}")
    else:
        print("No persisted embeddings found.")

    # Pick an image from dataset
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    if not files:
        print("No images in dataset to test.")
        return

    test_file = files[0]
    print(f"Testing with image: {test_file}")
    
    img_path = os.path.join(dataset_dir, test_file)
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image.")
        return
        
    # Simulate the pipeline in gen_frames
    # 1. Detect and Align
    # gen_frames uses mtcnn.detect on small image, then extracts from full image
    # We'll just use embedding_from_path logic which uses detect_and_align or extract
    
    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    print("Extracting embedding...")
    try:
        # We use the same method as in gen_frames: mtcnn.extract -> embedding_from_face_tensor
        mtcnn = model_utils.get_mtcnn()
        # Detect first
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None:
            print("No face detected in image.")
            return
            
        print(f"Detected {len(boxes)} faces.")
        box = boxes[0]
        
        # Extract
        face_tensors = mtcnn.extract(pil_img, [box], save_path=None)
        if face_tensors is None:
            print("Failed to extract face tensor.")
            return
            
        face_t = face_tensors[0]
        emb = model_utils.embedding_from_face_tensor(face_t)
        
        # 2. Match Persisted
        threshold = app.config.get('KNOWN_DISTANCE_THRESHOLD', 1.4)
        print(f"Matching with threshold {threshold}...")
        
        known, name, dist = match_persisted_embedding(emb, threshold)
        print(f"match_persisted_embedding result: known={known}, name={name}, dist={dist}")
        
        # 3. Is Known (Classifier)
        known_clf, name_clf, dist_clf = is_known(emb, threshold=threshold)
        print(f"is_known result: known={known_clf}, name={name_clf}, dist={dist_clf}")
        
    except Exception as e:
        print(f"Error during reproduction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
