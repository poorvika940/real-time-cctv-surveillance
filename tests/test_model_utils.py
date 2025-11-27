import os
import sys
import tempfile
import numpy as np

# ensure project root is on sys.path for imports when pytest runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_utils import train_knn, load_classifier, predict


def test_knn_train_and_predict():
    # Create a tiny synthetic dataset with 3 classes
    rng = np.random.RandomState(0)
    emb_dim = 128
    embeddings = []
    labels = []
    for cls in ['alice', 'bob', 'carol']:
        center = rng.randn(emb_dim)
        for i in range(3):
            embeddings.append(center + 0.1 * rng.randn(emb_dim))
            labels.append(cls)
    embeddings = np.stack(embeddings, axis=0)

    # Train
    clf = train_knn(embeddings, labels, n_neighbors=1)
    assert clf is not None

    # Pick a sample from alice and ensure predicted label is 'alice'
    sample = embeddings[0]
    res = predict(sample)
    assert res is not None
    assert 'label' in res
    assert res['label'] in labels
