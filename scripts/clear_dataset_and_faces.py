import os, shutil, sys
# ensure project root is on sys.path so imports work when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import SessionLocal, Face, init_db
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET = os.path.join(BASE, 'dataset')
MODEL_DIR = os.path.join(BASE, 'models')
print('Base:', BASE)
# remove all files under dataset
if os.path.exists(DATASET):
    removed_files = 0
    for root, dirs, files in os.walk(DATASET):
        for f in files:
            p = os.path.join(root, f)
            try:
                os.remove(p)
                removed_files += 1
            except Exception as e:
                print('failed remove', p, e)
    # remove empty subdirs
    for root, dirs, files in os.walk(DATASET, topdown=False):
        for d in dirs:
            dp = os.path.join(root, d)
            try:
                os.rmdir(dp)
            except Exception:
                pass
    print('Removed dataset files:', removed_files)
else:
    print('Dataset dir not found:', DATASET)

# clear Face rows
init_db()
session = SessionLocal()
count = session.query(Face).count()
print('Face rows before:', count)
try:
    session.query(Face).delete()
    session.commit()
    print('Deleted Face rows')
except Exception as e:
    print('Failed to delete Face rows', e)
finally:
    session.close()

# remove model artifacts
files_to_remove = ['classifier.joblib','knn_classifier.joblib','embeddings.npz','predictions.csv']
removed_models = []
for fn in files_to_remove:
    p = os.path.join(MODEL_DIR, fn)
    if os.path.exists(p):
        try:
            os.remove(p)
            removed_models.append(fn)
        except Exception as e:
            print('failed remove model file', p, e)
print('Removed model files:', removed_models)

# final DB count
session = SessionLocal()
print('Face rows after:', session.query(Face).count())
session.close()
print('Done')
