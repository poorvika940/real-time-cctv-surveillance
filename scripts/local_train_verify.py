import os, sys, numpy as np
from PIL import Image
from model_utils import detect_and_align, embedding_from_face_tensor, train_classifier, is_known, get_mtcnn

# pick a dataset image that exists
img_path = os.path.join('dataset', 'Vinuth', 'IMG-20251010-WA0016[1].jpg')
if not os.path.exists(img_path):
    print('Image not found:', img_path)
    sys.exit(2)

print('Loading image', img_path)
img = Image.open(img_path).convert('RGB')
# detect + align
faces = detect_and_align(img, device='cpu')
if not faces:
    print('No faces detected in image')
    sys.exit(3)
face_t = faces[0]
emb = embedding_from_face_tensor(face_t, device='cpu')
print('Embedding shape', emb.shape)
# Train classifier with a single sample (kneighbors=1)
label = 'LocalTestUser'
print('Training KNN with 1 sample (n_neighbors=1)')
obj = train_classifier(np.stack([emb], axis=0), [label], method='knn', n_neighbors=1)
print('Train returned:', obj.get('method'))
# Test is_known
known, lbl, dist = is_known(emb, threshold=0.9)
print('After training, is_known ->', known, lbl, dist)
if known:
    print('SUCCESS: face recognized as', lbl)
else:
    print('FAIL: still unknown')
