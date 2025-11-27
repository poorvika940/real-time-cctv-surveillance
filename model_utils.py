import os
import numpy as np
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
except Exception:
    # In environments without facenet-pytorch installed, fallback to None and raise at runtime usage
    MTCNN = None
    InceptionResnetV1 = None
    torch = None
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import __version__ as sklearn_version
import joblib

# Models will be loaded lazily
_mtcnn = None
_mtcnn_device = None
_resnet = None
_resnet_device = None
_classifier_path = os.path.join(os.path.dirname(__file__), 'models', 'classifier.joblib')


def available_device():
    """Return 'cuda' if CUDA is available, otherwise 'cpu'."""
    if torch is not None and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_mtcnn(device='cpu'):
    global _mtcnn, _mtcnn_device
    # normalize device string
    device = device or available_device()
    if _mtcnn is None or _mtcnn_device != device:
        if MTCNN is None:
            raise RuntimeError('facenet-pytorch is not available')
        _mtcnn = MTCNN(keep_all=True, device=device)
        _mtcnn_device = device
    return _mtcnn


def get_resnet(device='cpu'):
    global _resnet, _resnet_device
    device = device or available_device()
    if _resnet is None or _resnet_device != device:
        if InceptionResnetV1 is None:
            raise RuntimeError('facenet-pytorch is not available')
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        _resnet_device = device
    return _resnet


def detect_and_align(image, device='cpu'):
    """
    Given a PIL image or numpy array (H,W,3), detect faces and return aligned tensors for each face.
    Returns list of torch tensors (3,160,160) normalized for resnet.
    """
    mtcnn = get_mtcnn(device=device)
    if isinstance(image, np.ndarray):
        from PIL import Image
        image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
    faces = mtcnn(image)
    if faces is None:
        return []
    if isinstance(faces, torch.Tensor):
        faces = [faces]
    faces = [f.to(device) for f in faces]
    return faces


def embedding_from_face_tensor(face_tensor, device='cpu'):
    """face_tensor: torch.Tensor (3,160,160) - normalized by MTCNN already"""
    resnet = get_resnet(device=device)
    # Normalize input shapes so we always feed a tensor of shape (B,3,160,160) to the resnet.
    # facenet-pytorch can return tensors with extra batch dimensions in some versions or usage
    # (e.g. shapes like [1,1,3,160,160], [1,3,160,160], or already [3,160,160]).
    if torch is None:
        raise RuntimeError('torch is required for embedding computation')
    t = face_tensor
    # convert numpy input to torch if needed
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    # ensure tensor
    if not isinstance(t, torch.Tensor):
        raise RuntimeError('Unexpected face_tensor type: %s' % type(t))

    # collapse any unnecessary leading singleton dims, then ensure (B,3,H,W)
    try:
        # remove any singleton dims at the front until dim 0 == 3 or dim 1 == 3
        while t.dim() > 0 and t.shape[0] == 1 and t.dim() > 1 and t.shape[1] != 3:
            t = t.squeeze(0)
    except Exception:
        pass

    # Now handle common cases
    if t.dim() == 3:
        # (3,H,W) -> make batch
        inp = t.unsqueeze(0).to(device)
    elif t.dim() == 4:
        # (1,3,H,W) or (N,3,H,W) -> ok
        # if channels not in dim 1, try to move
        if t.shape[1] != 3 and t.shape[0] == 3:
            # for shape (3,1,H,W) or (3,H,W,1) attempt to permute
            try:
                inp = t.permute(1,0,2,3).to(device)
            except Exception:
                inp = t.to(device)
        else:
            inp = t.to(device)
    else:
        # fallback: try to squeeze then unsqueeze to get (3,H,W)
        try:
            tt = t.squeeze()
            if tt.dim() == 3:
                inp = tt.unsqueeze(0).to(device)
            elif tt.dim() == 4 and tt.shape[1] == 3:
                inp = tt.to(device)
            else:
                # last resort: reshape if possible
                inp = tt.view(-1, 3, 160, 160).to(device)
        except Exception:
            raise RuntimeError(f'unable to normalize face tensor shape: {t.shape}')

    with torch.no_grad():
        emb = resnet(inp)
    return emb.cpu().numpy().reshape(-1)


def compute_embeddings_for_images(image_tensors, device='cpu'):
    """image_tensors: iterable of torch.Tensor face tensors"""
    embs = [embedding_from_face_tensor(t, device=device) for t in image_tensors]
    return np.stack(embs, axis=0)


def save_classifier(obj):
    os.makedirs(os.path.dirname(_classifier_path), exist_ok=True)
    joblib.dump(obj, _classifier_path)


def load_classifier():
    if not os.path.exists(_classifier_path):
        return None
    obj = joblib.load(_classifier_path)
    try:
        if isinstance(obj, dict):
            saved_version = obj.get('sklearn_version')
            if saved_version and saved_version != sklearn_version:
                print(f"[classifier] sklearn version changed ({saved_version} -> {sklearn_version}); retraining required")
                return None
            if saved_version is None:
                # Old artifacts without metadata should be rebuilt so add_face/upload flows stay reliable
                print('[classifier] legacy classifier missing version metadata; retraining required')
                return None
        else:
            # legacy plain estimator saved directly; rebuild to ensure consistent metadata
            print('[classifier] legacy plain estimator detected; retraining required')
            return None
    except Exception:
        # If metadata inspection fails, force a rebuild to stay safe
        return None
    return obj


def train_knn(embeddings, labels, n_neighbors=3):
    """Train and persist a KNN classifier. Returns fitted classifier."""
    return train_classifier(embeddings, labels, method='knn', n_neighbors=n_neighbors)


def train_classifier(embeddings, labels, method='knn', n_neighbors=3, svm_C=1.0):
    """Train and persist a classifier. method: 'knn' or 'svm'. Returns a dict with metadata."""
    method = method.lower()
    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(embeddings, labels)
        obj = {'method': 'knn', 'clf': clf, 'sklearn_version': sklearn_version}
        save_classifier(obj)
        return obj
    elif method == 'svm':
        # probability=True to allow using predict_proba for scoring
        clf = SVC(C=svm_C, probability=True)
        clf.fit(embeddings, labels)
        obj = {'method': 'svm', 'clf': clf, 'sklearn_version': sklearn_version}
        save_classifier(obj)
        return obj
    else:
        raise ValueError(f'Unsupported classifier method: {method}')


def predict(embedding):
    """Return dict with label and either distance (for knn) or score/prob (for svm).
    Returns None if no classifier persisted.
    """
    obj = load_classifier()
    if obj is None:
        return None
    method = obj.get('method') if isinstance(obj, dict) else 'knn'
    clf = obj.get('clf') if isinstance(obj, dict) else obj
    if method == 'knn' and hasattr(clf, 'kneighbors'):
        dist, idx = clf.kneighbors([embedding])
        lbl = clf.predict([embedding])[0]
        return {'label': lbl, 'distance': float(dist[0][0])}
    if method == 'svm' and hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba([embedding])[0]
        # take the label with highest probability
        idx = int(np.argmax(probs))
        lbl = clf.classes_[idx]
        prob = float(probs[idx])
        # define a pseudo-distance as 1 - prob (smaller => more confident)
        return {'label': lbl, 'distance': 1.0 - prob, 'score': prob}
    # fallback: just predict label
    lbl = clf.predict([embedding])[0]
    return {'label': lbl, 'distance': None}


def is_known(embedding, threshold=0.8, prob_threshold=0.5):
    """Returns (known_bool, label, distance) comparing classifier output to a threshold.
    For KNN, 'distance' is returned and compared to threshold (smaller => more similar).
    For SVM, 'score' key is provided (probability) and compared to prob_threshold.
    """
    pred = predict(embedding)
    if pred is None:
        return (False, None, None)
    distance = pred.get('distance')
    score = pred.get('score')
    label = pred.get('label')
    if score is not None:
        # for probabilistic classifiers (SVM with probability=True)
        return (score >= prob_threshold, label, 1.0 - score)
    if distance is None:
        return (True, label, None)
    return (distance <= threshold, label, distance)
