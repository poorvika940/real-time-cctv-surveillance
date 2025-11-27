import sys
import os
import numpy as np
# ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import train_classifier, load_classifier, predict, is_known


def make_fake_embeddings(n_per_class=5, dim=128):
    # class A centered at +1, class B centered at -1
    a = np.random.randn(n_per_class, dim) * 0.1 + 1.0
    b = np.random.randn(n_per_class, dim) * 0.1 - 1.0
    X = np.vstack([a, b])
    y = ['A'] * n_per_class + ['B'] * n_per_class
    return X, y


def test_train_and_predict_knn(tmp_path):
    X, y = make_fake_embeddings(n_per_class=6, dim=64)
    obj = train_classifier(X, y, method='knn', n_neighbors=3)
    assert obj['method'] == 'knn'
    # pick a sample from class A and ensure predict returns A
    sample = X[0]
    p = predict(sample)
    assert p is not None and 'label' in p


def test_train_and_predict_svm(tmp_path):
    X, y = make_fake_embeddings(n_per_class=6, dim=64)
    obj = train_classifier(X, y, method='svm', svm_C=1.0)
    assert obj['method'] == 'svm'
    sample = X[0]
    p = predict(sample)
    assert p is not None and 'label' in p and ('score' in p or 'distance' in p)
