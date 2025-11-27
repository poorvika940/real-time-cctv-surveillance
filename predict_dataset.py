"""
Predict on images under dataset/ using the trained classifier saved in models/classifier.joblib
Produces CSV: models/predictions.csv with columns:
  image_path, true_label, predicted_label, distance, known

Run with project's venv python:
& ".\\.venv312\\Scripts\\python.exe" predict_dataset.py
"""
import os
import csv
from PIL import Image
import numpy as np

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, 'dataset')
MODEL_DIR = os.path.join(BASE, 'models')
OUT_CSV = os.path.join(MODEL_DIR, 'predictions.csv')

# add project root to path for imports
import sys
sys.path.insert(0, BASE)

from model_utils import get_mtcnn, embedding_from_face_tensor, predict, is_known, load_classifier

# check classifier exists
clf = load_classifier()
if clf is None:
    print('No classifier found at models/classifier.joblib. Run training first.')
    raise SystemExit(2)

mtcnn = None
try:
    mtcnn = get_mtcnn(device='cpu')
except Exception as e:
    print('Failed to initialize MTCNN:', e)
    raise

rows = []

# Walk dataset: if subdirs exist, treat subdir as true label, else infer label from filename prefix
for entry in sorted(os.listdir(DATASET_DIR)):
    p = os.path.join(DATASET_DIR, entry)
    if os.path.isdir(p):
        true_label = entry
        files = [f for f in sorted(os.listdir(p)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fn in files:
            path = os.path.join(p, fn)
            try:
                img = Image.open(path).convert('RGB')
                faces = mtcnn(img)
                if faces is None or len(faces) == 0:
                    rows.append((path, true_label, '', '', False))
                    print('No face detected in', path)
                    continue
                face_t = faces[0]
                emb = embedding_from_face_tensor(face_t, device='cpu')
                pred = predict(emb)
                if pred is None:
                    rows.append((path, true_label, '', '', False))
                else:
                    lbl = pred.get('label')
                    dist = pred.get('distance')
                    # determine known/unknown according to model_utils.is_known behaviour
                    known, _, _ = is_known(emb)
                    rows.append((path, true_label, lbl, dist if dist is not None else '', bool(known)))
                    print('Predicted', path, '->', lbl)
            except Exception as e:
                print('Failed', path, e)
                rows.append((path, true_label, '', '', False))
    else:
        # top-level file
        if not entry.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if '_' in entry:
            true_label = entry.split('_')[0]
        elif '-' in entry:
            true_label = entry.split('-')[0]
        else:
            true_label = 'unknown'
        path = p
        try:
            img = Image.open(path).convert('RGB')
            faces = mtcnn(img)
            if faces is None or len(faces) == 0:
                rows.append((path, true_label, '', '', False))
                print('No face detected in', path)
                continue
            face_t = faces[0]
            emb = embedding_from_face_tensor(face_t, device='cpu')
            pred = predict(emb)
            if pred is None:
                rows.append((path, true_label, '', '', False))
            else:
                lbl = pred.get('label')
                dist = pred.get('distance')
                known, _, _ = is_known(emb)
                rows.append((path, true_label, lbl, dist if dist is not None else '', bool(known)))
                print('Predicted', path, '->', lbl)
        except Exception as e:
            print('Failed', path, e)
            rows.append((path, true_label, '', '', False))

# write CSV
os.makedirs(MODEL_DIR, exist_ok=True)
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image_path', 'true_label', 'predicted_label', 'distance', 'known'])
    for r in rows:
        w.writerow(r)

print('Wrote predictions to', OUT_CSV)
