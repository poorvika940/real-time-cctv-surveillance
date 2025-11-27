import os
import sys
import numpy as np
from database import SessionLocal, Face
import model_utils

def check_system():
    print("Checking system state...")
    
    # Check DB
    session = SessionLocal()
    faces = session.query(Face).all()
    session.close()
    print(f"Database contains {len(faces)} faces:")
    for f in faces:
        print(f" - ID: {f.id}, Name: {f.name}, Image: {f.image_path}")

    # Check Classifier
    clf_obj = model_utils.load_classifier()
    if clf_obj is None:
        print("Classifier NOT found or could not be loaded.")
    else:
        clf = clf_obj.get('clf')
        method = clf_obj.get('method')
        print(f"Classifier loaded. Method: {method}")
        if hasattr(clf, 'classes_'):
            print(f"Classes: {clf.classes_}")
        else:
            print("Classifier has no classes_ attribute.")

    # Check Embeddings file
    embs, labels = model_utils._load_persisted_embeddings()
    if embs is not None:
        print(f"Persisted embeddings found: {len(labels)} entries.")
    else:
        print("No persisted embeddings file found.")

if __name__ == "__main__":
    check_system()
