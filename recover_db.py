import os
import sys
from database import SessionLocal, Face
from app import retrain_classifier

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

def recover_db():
    print("Recovering database from dataset images...")
    session = SessionLocal()
    
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images.")
    
    added_count = 0
    for filename in files:
        # Parse name from filename: Name_Timestamp.jpg
        # We assume the last underscore separates name and timestamp
        try:
            name_part = filename.rsplit('_', 1)[0]
            # Handle cases where name might have underscores, e.g. Poorvika_D_M
            # The format seems to be Name_Timestamp.jpg
            name = name_part
            
            # Check if exists
            full_path = os.path.join(DATASET_DIR, filename)
            exists = session.query(Face).filter_by(image_path=full_path).first()
            
            if not exists:
                face = Face(name=name, image_path=full_path, profession='Recovered')
                session.add(face)
                added_count += 1
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            
    session.commit()
    session.close()
    print(f"Added {added_count} faces to the database.")
    
    if added_count > 0:
        print("Retraining classifier...")
        retrain_classifier(device='cpu', method='knn')
        print("Recovery complete.")
    else:
        print("No new faces added. Retraining anyway to ensure classifier is up to date.")
        retrain_classifier(device='cpu', method='knn')

if __name__ == "__main__":
    recover_db()
