import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import detect_and_align, embedding_from_face_tensor, is_known

from PIL import Image

img = sys.argv[1] if len(sys.argv)>1 else 'dataset/Poorvika_1762692866.jpg'
print('Checking image', img)
if not os.path.exists(img):
    print('file not found', img); sys.exit(2)
im = Image.open(img).convert('RGB')
try:
    faces = detect_and_align(im, device='cpu')
except Exception as e:
    print('detect_and_align error:', e)
    faces = []
print('faces found:', len(faces))
if not faces:
    print('No faces detected')
    sys.exit(2)
for i,f in enumerate(faces):
    try:
        emb = embedding_from_face_tensor(f, device='cpu')
    except Exception as e:
        print('embedding error', e)
        continue
    known,label,d = is_known(emb)
    print('face', i, 'known=', known, 'label=', label, 'distance=', d)

print('done')
