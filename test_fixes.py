"""
Test script to verify notification queue and face recognition fixes.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_notification_queue():
    """Test that notification queue is working."""
    print("=" * 60)
    print("Testing Notification Queue")
    print("=" * 60)
    
    from app import NOTIFICATION_QUEUE, _notification_worker_thread, _notification_worker_running
    
    print(f"Notification queue initialized: {NOTIFICATION_QUEUE is not None}")
    print(f"Worker thread exists: {_notification_worker_thread is not None}")
    print(f"Worker thread running: {_notification_worker_running}")
    
    if _notification_worker_thread:
        print(f"Worker thread alive: {_notification_worker_thread.is_alive()}")
        print(f"Worker thread name: {_notification_worker_thread.name}")
    
    print(f"Queue size: {NOTIFICATION_QUEUE.qsize()}")
    print()

def test_threshold_config():
    """Test that threshold is correctly configured."""
    print("=" * 60)
    print("Testing Threshold Configuration")
    print("=" * 60)
    
    from app import app
    
    threshold = app.config.get('KNOWN_DISTANCE_THRESHOLD')
    print(f"KNOWN_DISTANCE_THRESHOLD: {threshold}")
    print(f"Expected: 1.2")
    print(f"Match: {threshold == 1.2}")
    print()

def test_face_recognition():
    """Test face recognition with existing data."""
    print("=" * 60)
    print("Testing Face Recognition")
    print("=" * 60)
    
    from app import app, match_persisted_embedding, _load_persisted_embeddings
    import model_utils
    
    # Check if classifier is loaded
    clf = model_utils.load_classifier()
    if clf:
        print(f"✓ Classifier loaded: {clf.get('method')}")
    else:
        print("✗ Classifier NOT loaded")
    
    # Check persisted embeddings
    embs, labels = _load_persisted_embeddings()
    if embs is not None:
        print(f"✓ Persisted embeddings loaded: {len(labels)} faces")
        print(f"  Labels: {labels}")
    else:
        print("✗ No persisted embeddings found")
    
    # Check database
    from database import SessionLocal, Face
    session = SessionLocal()
    faces = session.query(Face).all()
    session.close()
    
    print(f"✓ Database contains {len(faces)} faces:")
    for f in faces:
        print(f"  - {f.name} (ID: {f.id})")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FACE RECOGNITION & NOTIFICATION QUEUE TEST")
    print("=" * 60 + "\n")
    
    try:
        test_threshold_config()
        test_notification_queue()
        test_face_recognition()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
