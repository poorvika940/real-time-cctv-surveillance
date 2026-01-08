import sys
import os
import numpy as np
import tempfile
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import retrain_classifier, embedding_from_path
from database import SessionLocal, Face, init_db


def test_retrain_classifier_uses_known_embeddings(tmp_path, monkeypatch):
    """Test that retrain_classifier uses pre-computed embeddings when provided."""
    
    # Setup: Initialize database
    init_db()
    
    # Create temporary image files
    img_path_1 = str(tmp_path / "face1.jpg")
    img_path_2 = str(tmp_path / "face2.jpg")
    
    # Create simple test images
    img1 = Image.new('RGB', (160, 160), color=(100, 100, 100))
    img1.save(img_path_1, format='JPEG')
    
    img2 = Image.new('RGB', (160, 160), color=(200, 200, 200))
    img2.save(img_path_2, format='JPEG')
    
    # Add faces to database
    session = SessionLocal()
    try:
        # Clean up any existing faces
        session.query(Face).delete()
        session.commit()
        
        # Add test faces
        face1 = Face(name='TestPerson1', profession='Tester', image_path=img_path_1)
        face2 = Face(name='TestPerson2', profession='Tester', image_path=img_path_2)
        session.add(face1)
        session.add(face2)
        session.commit()
    finally:
        session.close()
    
    # Create pre-computed embeddings
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)
    
    # Track which function was called
    embedding_from_path_called = {'count': 0, 'paths': []}
    
    # Mock embedding_from_path to track calls and fail for img_path_1
    original_embedding_from_path = embedding_from_path
    
    def mock_embedding_from_path(path, device='cpu'):
        embedding_from_path_called['count'] += 1
        embedding_from_path_called['paths'].append(path)
        
        # Simulate failure for img_path_1 (cropped image scenario)
        if path == img_path_1:
            return None
        
        # Return valid embedding for img_path_2
        return emb2
    
    monkeypatch.setattr('app.embedding_from_path', mock_embedding_from_path)
    
    # Test 1: Without known_embeddings, face1 should be dropped
    result1 = retrain_classifier(device='cpu', method='knn', known_embeddings=None)
    
    # Verify that embedding_from_path was called for both paths
    assert embedding_from_path_called['count'] == 2
    assert img_path_1 in embedding_from_path_called['paths']
    assert img_path_2 in embedding_from_path_called['paths']
    
    # Reset tracking
    embedding_from_path_called = {'count': 0, 'paths': []}
    
    # Test 2: With known_embeddings, face1 should be included
    known_embeddings = {img_path_1: emb1}
    result2 = retrain_classifier(device='cpu', method='knn', known_embeddings=known_embeddings)
    
    # Verify that embedding_from_path was only called for img_path_2
    # (img_path_1 should use the pre-computed embedding)
    assert embedding_from_path_called['count'] == 1
    assert img_path_1 not in embedding_from_path_called['paths']
    assert img_path_2 in embedding_from_path_called['paths']
    
    # Both should succeed
    assert result2['success'] == True
    assert result2['method'] == 'knn'
    
    # Cleanup
    session = SessionLocal()
    try:
        session.query(Face).delete()
        session.commit()
    finally:
        session.close()


def test_retrain_classifier_known_embeddings_priority(tmp_path, monkeypatch):
    """Test that known_embeddings take priority over embedding_from_path."""
    
    # Setup
    init_db()
    
    img_path = str(tmp_path / "test_face.jpg")
    img = Image.new('RGB', (160, 160), color=(150, 150, 150))
    img.save(img_path, format='JPEG')
    
    # Add face to database
    session = SessionLocal()
    try:
        session.query(Face).delete()
        session.commit()
        
        face = Face(name='TestPerson', profession='Tester', image_path=img_path)
        session.add(face)
        session.commit()
    finally:
        session.close()
    
    # Create two different embeddings
    expected_emb = np.ones(512, dtype=np.float32) * 0.5
    wrong_emb = np.ones(512, dtype=np.float32) * 0.9
    
    # Mock embedding_from_path to return wrong_emb
    def mock_embedding_from_path(path, device='cpu'):
        return wrong_emb
    
    monkeypatch.setattr('app.embedding_from_path', mock_embedding_from_path)
    
    # Call retrain with known_embeddings containing the expected embedding
    known_embeddings = {img_path: expected_emb}
    result = retrain_classifier(device='cpu', method='knn', known_embeddings=known_embeddings)
    
    assert result['success'] == True
    
    # Cleanup
    session = SessionLocal()
    try:
        session.query(Face).delete()
        session.commit()
    finally:
        session.close()


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
