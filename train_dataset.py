"""
Quick training script: walks `dataset/` where subfolders are class labels or images named with label prefix.
It will detect faces, compute embeddings via model_utils, train a KNN classifier, and save it to `models/classifier.joblib`.
This script is conservative: it uses CPU and skips images where no face is detected.
Run from project root: python train_dataset.py
"""
import os
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_utils import get_mtcnn, embedding_from_face_tensor, train_knn, save_classifier

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, 'dataset')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

device = 'cpu'
mtcnn = None
try:
    mtcnn = get_mtcnn(device=device)
except Exception as e:
    print('Failed to initialize MTCNN:', e)
    print('Ensure facenet-pytorch is installed and available in this environment.')
    sys.exit(1)

embs = []
labels = []

# If dataset contains subdirectories, treat each subdir name as label and images inside as examples
for entry in sorted(os.listdir(DATASET_DIR)):
    p = os.path.join(DATASET_DIR, entry)
    if os.path.isdir(p):
        label = entry
        print('Processing directory label:', label)
        for fn in sorted(os.listdir(p)):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(p, fn)
            try:
                img = Image.open(path).convert('RGB')
                faces = mtcnn(img)
                if faces is None:
                    print('  warning: no face detected in', path)
                    continue
                if hasattr(faces, 'unsqueeze'):
                    # single tensor or batch
                    if faces.ndim == 3:
                        face_t = faces
                    else:
                        face_t = faces[0]
                else:
                    face_t = faces[0]
                emb = embedding_from_face_tensor(face_t, device=device)
                embs.append(emb)
                labels.append(label)
                print('  added', path)
            except Exception as e:
                print('  failed', path, e)
    else:
        # file at top-level - try to infer label from filename prefix (label_rest.jpg)
        if not entry.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        # find a label by splitting on underscore or hyphen
        if '_' in entry:
            label = entry.split('_')[0]
        elif '-' in entry:
            label = entry.split('-')[0]
        else:
            label = 'unknown'
        path = p
        try:
            img = Image.open(path).convert('RGB')
            faces = mtcnn(img)
            if faces is None:
                print('  warning: no face detected in', path)
                continue
            face_t = faces[0] if len(faces) > 0 else faces
            emb = embedding_from_face_tensor(face_t, device=device)
            embs.append(emb)
            labels.append(label)
            print('  added', path, '->', label)
        except Exception as e:
            print('  failed', path, e)

if len(embs) == 0:
    print('No embeddings extracted. Training aborted.')
    sys.exit(2)

embs = np.stack(embs, axis=0)
print('Collected embeddings:', embs.shape, 'labels count:', len(labels))

obj = train_knn(embs, labels, n_neighbors=3)
print('Training complete. Method:', obj.get('method') if isinstance(obj, dict) else 'knn')
print('Classifier saved to models/classifier.joblib')

# print class distribution
from collections import Counter
print('Class counts:', Counter(labels))

print('Done')
