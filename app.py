# ...existing code...

import os
import io
import cv2
import time
import base64
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
from model_utils import detect_and_align, embedding_from_face_tensor, compute_embeddings_for_images, predict, train_knn, load_classifier, save_classifier, is_known, get_mtcnn, available_device
from database import init_db, SessionLocal, Face, AlertAck
from database import AlertAck
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from flask_socketio import SocketIO
from tracker import SimpleTracker
import threading
import queue
from flask import current_app
# LLM services
try:
    from services.llm import generate as llm_generate, get_default_llm, set_default_llm
except Exception:
    # In case services package is not available, provide a fallback
    def llm_generate(prompt, model=None, **opts):
        return {'success': True, 'model': model or 'none', 'text': '[llm-not-configured] ' + str(prompt)[:200]}
    def get_default_llm():
        return None
    def set_default_llm(m):
        return m


# performance stats
PERF_STATS = {'detect_total': 0.0, 'embed_total': 0.0, 'frames': 0}

# Last camera error for admin/debug UI
LAST_CAMERA_ERROR = {'msg': None, 'ts': None}

# trackers per camera id
TRACKERS = {}
# short-term cache of recently-added embeddings to suppress duplicate add-face prompts
# entries: {'emb': np.ndarray, 'label': str, 'ts': float}
RECENT_EMBED_CACHE = []
# seconds to keep an embedding in recent cache
RECENT_EMBED_TTL = 300.0
# recent alerts store (keeps last N alerts so admin UI can show thumbnails)
# will be loaded from disk on startup
ALERTS = []
# per-camera settings (show_ids, iou_threshold, max_missed, alert_cooldown)
TRACKER_SETTINGS = {}
# fallback alert cooldown when tracker info missing (per camera)
NO_TRACK_ALERT_COOLDOWN = 5.0
# ensure only one unknown-face notification per identity until it becomes known or is acknowledged
ALERT_LOCKS = {}

# in-memory map of acknowledged keys (camera::label or camera::track) -> last ack timestamp
ACK_KEYS = {}

# Notification queue for sequential processing
NOTIFICATION_QUEUE = queue.Queue()
NOTIFICATION_DELAY = 2.0  # seconds between notifications
_notification_worker_thread = None
_notification_worker_running = False

# persisted embeddings path for incremental updates (defined after MODEL_DIR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret'
# Threshold for declaring a face known (distance units depend on embedding normalization)
# Set to 1.0 for strict matching to prevent unknown faces from matching known faces
app.config['KNOWN_DISTANCE_THRESHOLD'] = 1.0
# Default pipeline tuning knobs (overridable via query params)
app.config['DEFAULT_FRAME_SKIP'] = int(os.environ.get('FRAME_SKIP_DEFAULT', 2))
app.config['DEFAULT_DEVICE'] = os.environ.get('DEFAULT_DEVICE') or available_device()
# Motion gating removed per request; always detect every scheduled frame
BASE_DIR = os.path.dirname(__file__)
app.config.setdefault('NO_TRACK_ALERT_COOLDOWN', float(os.environ.get('NO_TRACK_ALERT_COOLDOWN', 5.0)))
app.config.setdefault('ALERT_REARM_SECONDS', float(os.environ.get('ALERT_REARM_SECONDS', 300.0)))
app.config.setdefault('ACK_EXPIRY_SECONDS', float(os.environ.get('ACK_EXPIRY_SECONDS', 900.0)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'snapshots')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
FACE_EMBED_DIR = os.path.join(MODEL_DIR, 'face_vectors')
os.makedirs(FACE_EMBED_DIR, exist_ok=True)

# persisted embeddings path for incremental updates
_EMBED_PATH = os.path.join(MODEL_DIR, 'embeddings.npz')

# cache for persisted embeddings (npz) to avoid reloading on every frame
_PERSISTED_EMBED_CACHE = {
    'mtime': None,
    'embs': None,
    'labels': None,
}

# runtime LLM default (can be toggled at runtime via /admin/set_llm)
_RUNTIME_LLM = os.environ.get('DEFAULT_LLM', get_default_llm() or 'claude-sonnet-4.5')


# tracker settings persistence (file lives in models/)
_TRACKER_SETTINGS_PATH = os.path.join(MODEL_DIR, 'tracker_settings.json')

import warnings
import json
# suppress known Werkzeug/ast deprecation warnings (Python 3.12) during test/dev runs
warnings.filterwarnings("ignore", message=".*ast.Str is deprecated.*", category=DeprecationWarning)


def _quantize_box(box):
    try:
        if not box:
            return 'anon'
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return f"{cx//40}_{cy//40}_{w//40}_{h//40}"
    except Exception:
        return 'anon'


def _alert_key(camera_id, track_id=None, label=None, box=None):
    cam = str(camera_id)
    if track_id is not None:
        return f"{cam}::track::{track_id}"
    if label and str(label).strip().lower() not in ('unknown', ''):
        return f"{cam}::label::{str(label).strip().lower()}"
    if box is not None:
        bucket = _quantize_box(box)
        return f"{cam}::box::{bucket}"
    return f"{cam}::unknown"


def mark_unknown_alert(camera_id, track_id=None, label=None, box=None):
    """Return True if we should emit an alert, guaranteeing single notification per identity."""
    try:
        key = _alert_key(camera_id, track_id=track_id, label=label, box=box)
        now = time.time()
        rearm = float(app.config.get('ALERT_REARM_SECONDS', 300.0))
        entry = ALERT_LOCKS.get(key)
        if entry and (now - entry.get('ts', 0)) < rearm:
            return False
        ALERT_LOCKS[key] = {'ts': now}
        return True
    except Exception:
        return True


def release_alert_lock(camera_id, track_id=None, label=None):
    """Clear any dedupe locks for this identity so it can alert again later."""
    try:
        removed = False
        if track_id is not None:
            key = _alert_key(camera_id, track_id=track_id)
            removed = ALERT_LOCKS.pop(key, None) is not None or removed
        if label:
            key = _alert_key(camera_id, label=label)
            removed = ALERT_LOCKS.pop(key, None) is not None or removed
        return removed
    except Exception:
        return False


def _ack_expiry_seconds():
    try:
        return float(app.config.get('ACK_EXPIRY_SECONDS', 900.0))
    except Exception:
        return 900.0


def _is_embedding_path(path):
    if not path:
        return False
    lower = str(path).lower()
    return lower.endswith('.npy') or lower.endswith('.npz')


def _load_embedding_file(path):
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if 'emb' in arr.files:
            data = arr['emb']
        elif arr.files:
            data = arr[arr.files[0]]
        else:
            return None
    else:
        data = arr
    return np.asarray(data, dtype=float)


def _extract_face_tensor(img, device='cpu'):
    """Return a single aligned face tensor for the given PIL image."""
    tensors = detect_and_align(img, device=device)
    if not tensors:
        try:
            w, h = img.size
            mt = get_mtcnn(device=device)
            tensors = mt.extract(img, [(0, 0, w, h)], save_path=None) if mt else []
            if isinstance(tensors, torch.Tensor):
                tensors = [tensors]
        except Exception:
            tensors = []
    if not tensors:
        return None
    return tensors[0] if isinstance(tensors, (list, tuple)) else tensors


def _embedding_from_image(img, device='cpu'):
    face_t = _extract_face_tensor(img, device=device)
    if face_t is None:
        return None
    return embedding_from_face_tensor(face_t, device=device)


def _face_tensor_to_image(face_tensor):
    """Convert a normalized face tensor (3x160x160) to a PIL Image."""
    if face_tensor is None:
        return None
    try:
        if isinstance(face_tensor, torch.Tensor):
            arr = face_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(face_tensor)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        # tensors are normalized to [-1, 1]; map back to [0, 255]
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        return None


def embedding_from_path(path, device='cpu'):
    if not path:
        return None
    try:
        if _is_embedding_path(path):
            return _load_embedding_file(path)
        with Image.open(path).convert('RGB') as img:
            return _embedding_from_image(img, device=device)
    except Exception as exc:
        print('embedding_from_path failed for', path, exc)
        return None


def save_embedding_vector(emb, label):
    safe = ''.join(ch for ch in (label or 'face') if ch.isalnum() or ch in ('_', '-')) or 'face'
    ts = int(time.time())
    path = os.path.join(FACE_EMBED_DIR, f'{safe}_{ts}.npy')
    try:
        np.save(path, np.asarray(emb, dtype=np.float32))
    except Exception as exc:
        print('save_embedding_vector failed', exc)
        raise
    return path


def _ack_is_active(key):
    if not key:
        return False
    ts = ACK_KEYS.get(key)
    if not ts:
        return False
    if (time.time() - ts) > _ack_expiry_seconds():
        try:
            ACK_KEYS.pop(key, None)
        except Exception:
            pass
        return False
    return True


def remember_ack_key(key, ts=None):
    if not key:
        return
    try:
        ACK_KEYS[key] = float(ts if ts is not None else time.time())
    except Exception:
        ACK_KEYS[key] = time.time()


def clear_ack_key(key):
    if not key:
        return
    try:
        ACK_KEYS.pop(key, None)
    except Exception:
        pass


def load_tracker_settings():
    global TRACKER_SETTINGS
    try:
        if os.path.exists(_TRACKER_SETTINGS_PATH):
            with open(_TRACKER_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                TRACKER_SETTINGS = {str(k): v for k, v in (data or {}).items()}
                print(f"Loaded tracker settings from {_TRACKER_SETTINGS_PATH}")
                return
    except Exception as e:
        print('Failed to load tracker settings:', e)
    TRACKER_SETTINGS = {}


def save_tracker_settings():
    try:
        os.makedirs(os.path.dirname(_TRACKER_SETTINGS_PATH), exist_ok=True)
        with open(_TRACKER_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(TRACKER_SETTINGS, f, indent=2)
    except Exception as e:
        print('Failed to save tracker settings:', e)


_ALERTS_PATH = os.path.join(MODEL_DIR, 'alerts.json')
_VAPID_PATH = os.path.join(MODEL_DIR, 'vapid.json')
_PUSH_SUBS_PATH = os.path.join(MODEL_DIR, 'push_subscriptions.json')


def load_alerts():
    global ALERTS
    try:
        if os.path.exists(_ALERTS_PATH):
            with open(_ALERTS_PATH, 'r', encoding='utf-8') as f:
                ALERTS = json.load(f) or []
                print(f'Loaded {len(ALERTS)} alerts from {_ALERTS_PATH}')
                return
    except Exception as e:
        print('Failed to load alerts:', e)
    ALERTS = []


def save_alerts():
    try:
        os.makedirs(os.path.dirname(_ALERTS_PATH), exist_ok=True)
        with open(_ALERTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(ALERTS, f, indent=2)
    except Exception as e:
        print('Failed to save alerts:', e)


def load_vapid_keys():
    try:
        if os.path.exists(_VAPID_PATH):
            with open(_VAPID_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

@app.route('/rotate_vapid', methods=['POST'])
def rotate_vapid():
    """Regenerate VAPID key pair and clear existing push subscriptions so clients must re-subscribe.
    Returns new publicKey. Intended for development reset when notifications stop working.
    """
    try:
        vapid = generate_vapid_keys()
        # clear push subscriptions so browsers will re-subscribe with fresh keys
        try:
            if os.path.exists(_PUSH_SUBS_PATH):
                os.remove(_PUSH_SUBS_PATH)
        except Exception:
            pass
        return jsonify({'success': True, 'publicKey': vapid.get('publicKey'), 'rotated': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_vapid_keys():
    """Generate a P-256 VAPID keypair (base64url) and save to models/vapid.json.
    Returns dict with 'publicKey' and 'privateKey'. Requires `cryptography` package.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
    except Exception as e:
        raise RuntimeError('cryptography package is required to generate VAPID keys')
    # generate private key (P-256)
    priv = ec.generate_private_key(ec.SECP256R1(), default_backend())
    priv_numbers = priv.private_numbers()
    priv_value = priv_numbers.private_value
    priv_bytes = priv_value.to_bytes(32, 'big')
    # public key uncompressed format (0x04 | X | Y)
    pub = priv.public_key()
    pub_numbers = pub.public_numbers()
    x = pub_numbers.x.to_bytes(32, 'big')
    y = pub_numbers.y.to_bytes(32, 'big')
    uncompressed = b'\x04' + x + y
    import base64 as _b64
    public_key_b64 = _b64.urlsafe_b64encode(uncompressed).rstrip(b'=').decode('utf-8')
    private_key_b64 = _b64.urlsafe_b64encode(priv_bytes).rstrip(b'=').decode('utf-8')
    vapid = {'publicKey': public_key_b64, 'privateKey': private_key_b64}
    try:
        os.makedirs(os.path.dirname(_VAPID_PATH), exist_ok=True)
        with open(_VAPID_PATH, 'w', encoding='utf-8') as f:
            json.dump(vapid, f, indent=2)
    except Exception:
        pass
    return vapid


# Admin endpoint to get or set the runtime LLM provider
@app.route('/admin/get_llm', methods=['GET'])
def admin_get_llm():
    try:
        return jsonify({'success': True, 'llm': _RUNTIME_LLM})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/admin/set_llm', methods=['POST'])
def admin_set_llm():
    global _RUNTIME_LLM
    model = request.form.get('model') or request.json.get('model') if request.is_json else request.form.get('model')
    if not model:
        return jsonify({'success': False, 'error': 'missing model parameter'}), 400
    try:
        _RUNTIME_LLM = set_default_llm(model)
        return jsonify({'success': True, 'llm': _RUNTIME_LLM})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        print('Failed to load vapid keys:', e)
    return None


def _load_push_subscriptions():
    try:
        if os.path.exists(_PUSH_SUBS_PATH):
            with open(_PUSH_SUBS_PATH, 'r', encoding='utf-8') as f:
                subs = json.load(f) or []
                # filter out unsupported endpoints (e.g., WNS) automatically
                filtered = []
                changed = False
                for s in subs:
                    ep = str(s.get('endpoint') or '').lower()
                    if 'notify.windows.com' in ep:
                        changed = True
                        continue
                    filtered.append(s)
                if changed:
                    _save_push_subscriptions(filtered)
                return filtered
    except Exception as e:
        print('Failed to load push subscriptions:', e)
    return []


def _save_push_subscriptions(subs):
    try:
        os.makedirs(os.path.dirname(_PUSH_SUBS_PATH), exist_ok=True)
        with open(_PUSH_SUBS_PATH, 'w', encoding='utf-8') as f:
            json.dump(subs, f, indent=2)
    except Exception as e:
        print('Failed to save push subscriptions:', e)


def send_web_push(subscription, payload):
    endpoint = None
    try:
        # pywebpush is optional; this will fail gracefully if not installed
        from pywebpush import webpush, WebPushException
        vapid = load_vapid_keys()
        if not vapid or not vapid.get('publicKey') or not vapid.get('privateKey'):
            raise RuntimeError('VAPID keys not configured. Create models/vapid.json with publicKey/privateKey.')
        endpoint = subscription.get('endpoint') if isinstance(subscription, dict) else None
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=vapid.get('privateKey'),
            vapid_claims={"sub": "mailto:admin@example.com"}
        )
        if endpoint:
            print(f"[push] delivered -> {endpoint[:60]}")
        else:
            print('[push] delivered -> <unknown endpoint>')
        return True
    except Exception as e:
        if endpoint:
            print(f"[push] failed -> {endpoint[:60]} : {e}")
        else:
            print('send_web_push failed', e)
        return False


def notification_worker():
    """Background thread to process notifications sequentially with delays."""
    global _notification_worker_running
    _notification_worker_running = True
    print('[notification_worker] Started')
    
    while _notification_worker_running:
        try:
            # Wait for notification with timeout to allow checking running flag
            try:
                item = NOTIFICATION_QUEUE.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if item is None:  # Shutdown signal
                break
            
            # Send notification to all subscriptions
            subscriptions = item.get('subscriptions', [])
            payload = item.get('payload', {})
            
            print(f'[notification_worker] Processing notification: {payload.get("title", "Unknown")}')
            
            for sub in subscriptions:
                try:
                    send_web_push(sub, payload)
                except Exception as e:
                    print(f'[notification_worker] Failed to send to subscription: {e}')
            
            # Mark task as done
            NOTIFICATION_QUEUE.task_done()
            
            # Wait before processing next notification
            if not NOTIFICATION_QUEUE.empty():
                print(f'[notification_worker] Waiting {NOTIFICATION_DELAY}s before next notification')
                time.sleep(NOTIFICATION_DELAY)
                
        except Exception as e:
            print(f'[notification_worker] Error: {e}')
            try:
                NOTIFICATION_QUEUE.task_done()
            except Exception:
                pass
    
    print('[notification_worker] Stopped')


def queue_notification(subscriptions, payload):
    """Add a notification to the queue for sequential processing."""
    try:
        NOTIFICATION_QUEUE.put({
            'subscriptions': subscriptions,
            'payload': payload
        })
        print(f'[queue_notification] Queued notification: {payload.get("title", "Unknown")} (queue size: {NOTIFICATION_QUEUE.qsize()})')
        return True
    except Exception as e:
        print(f'[queue_notification] Failed to queue notification: {e}')
        return False


def start_notification_worker():
    """Start the notification worker thread if not already running."""
    global _notification_worker_thread, _notification_worker_running
    
    if _notification_worker_thread is not None and _notification_worker_thread.is_alive():
        return
    
    _notification_worker_running = True
    _notification_worker_thread = threading.Thread(
        target=notification_worker,
        daemon=True,
        name='NotificationWorker'
    )
    _notification_worker_thread.start()
    print('[start_notification_worker] Notification worker thread started')


@app.route('/sw.js')
def service_worker():
    try:
        return send_from_directory(os.path.join(app.root_path, 'static'), 'sw.js', mimetype='application/javascript')
    except Exception:
        return ('', 404)


def retrain_classifier(device='cpu', method='knn'):
    """Recompute embeddings from DB and train classifier. Runs in-process; safe to call in a background thread."""
    try:
        print('Starting retrain_classifier...')
        session = SessionLocal()
        faces = session.query(Face).all()
        session.close()
        if not faces:
            print('retrain_classifier: no faces found')
            return {'success': False, 'error': 'no faces'}
        embs = []
        labels = []
        for f in faces:
            try:
                emb = embedding_from_path(f.image_path, device=device)
                if emb is None:
                    continue
                embs.append(emb)
                labels.append(f.name)
            except Exception as e:
                print('retrain_classifier: failed processing', f.image_path, e)
                continue
        if not embs:
            print('retrain_classifier: no embeddings extracted')
            return {'success': False, 'error': 'no embeddings'}
        embs = np.stack(embs, axis=0)
        if method == 'knn':
            obj = train_knn(embs, labels, n_neighbors=3)
        else:
            from model_utils import train_classifier
            obj = train_classifier(embs, labels, method='svm', svm_C=1.0)
        print('Retrain complete, method=', obj.get('method') if isinstance(obj, dict) else 'knn')
        # Notify connected clients that retraining completed so UI can refresh state
        try:
            socketio.emit('retrain_complete', {'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn'}, namespace='/alerts')
        except Exception:
            pass
        return {'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn'}
    except Exception as e:
        print('retrain_classifier failed', e)
        try:
            socketio.emit('retrain_complete', {'success': False, 'error': str(e)}, namespace='/alerts')
        except Exception:
            pass
        return {'success': False, 'error': str(e)}


def _load_persisted_embeddings():
    """Load persisted embeddings and labels from disk if present.
    Returns (embs: np.ndarray (N, D), labels: list[str]) or (None, None) if not present.
    """
    try:
        if os.path.exists(_EMBED_PATH):
            data = np.load(_EMBED_PATH, allow_pickle=True)
            raw_embs = data.get('embs')
            raw_labels = data.get('labels')
            labels = raw_labels.tolist() if raw_labels is not None else []
            embs, labels = _sanitize_persisted_embeddings(raw_embs, labels)
            if embs is None or not len(embs):
                return None, None
            return embs, labels
    except Exception as e:
        print('_load_persisted_embeddings failed', e)
    return None, None


def _sanitize_persisted_embeddings(raw_embs, labels):
    """Ensure persisted embeddings form a 2D float array and drop inconsistent rows."""
    if raw_embs is None:
        return None, []
    arr = np.asarray(raw_embs)
    lbls = list(labels or [])
    if arr.dtype != object and arr.ndim == 2:
        arr = np.asarray(arr, dtype=float)
        # trim/extend labels to match row count
        if len(lbls) != arr.shape[0]:
            lbls = lbls[:arr.shape[0]]
        return arr, lbls

    # Handle ragged/object arrays by keeping only the dominant dimensionality
    rows = []
    dim_counts = {}
    flattened = arr.reshape(-1) if arr.ndim > 1 else arr
    for idx, item in enumerate(flattened):
        vec = np.asarray(item, dtype=float).reshape(-1)
        if vec.size == 0:
            continue
        dim_counts[vec.size] = dim_counts.get(vec.size, 0) + 1
        rows.append((idx, vec))
    if not rows:
        return None, []
    target_dim = max(dim_counts.items(), key=lambda kv: kv[1])[0]
    filtered = []
    filtered_labels = []
    for idx, vec in rows:
        if vec.size != target_dim:
            print(f"_load_persisted_embeddings: skipping vector #{idx} with dim {vec.size} (expect {target_dim})")
            continue
        filtered.append(vec)
        if idx < len(lbls):
            filtered_labels.append(lbls[idx])
    if not filtered:
        return None, []
    return np.stack(filtered, axis=0), filtered_labels


def _save_persisted_embeddings(embs, labels):
    try:
        os.makedirs(os.path.dirname(_EMBED_PATH), exist_ok=True)
        # ensure numpy array and object list
        np.savez_compressed(_EMBED_PATH, embs=np.asarray(embs), labels=np.asarray(labels, dtype=object))
        _invalidate_persisted_embeddings_cache()
        return True
    except Exception as e:
        print('_save_persisted_embeddings failed', e)
        return False


def _invalidate_persisted_embeddings_cache():
    global _PERSISTED_EMBED_CACHE
    _PERSISTED_EMBED_CACHE = {'mtime': None, 'embs': None, 'labels': None}


def _get_persisted_embeddings_cached():
    try:
        stat = os.stat(_EMBED_PATH)
        mtime = stat.st_mtime
    except OSError:
        return None, None
    global _PERSISTED_EMBED_CACHE
    if _PERSISTED_EMBED_CACHE['embs'] is None or _PERSISTED_EMBED_CACHE['mtime'] != mtime:
        embs, labels = _load_persisted_embeddings()
        if embs is None or labels is None or len(labels) == 0:
            return None, None
        _PERSISTED_EMBED_CACHE = {'mtime': mtime, 'embs': embs, 'labels': labels}
    return _PERSISTED_EMBED_CACHE['embs'], _PERSISTED_EMBED_CACHE['labels']


def match_persisted_embedding(embedding, threshold):
    try:
        embs, labels = _get_persisted_embeddings_cached()
        if embs is None or labels is None or len(labels) == 0:
            return False, None, None
        dists = np.linalg.norm(embs - embedding, axis=1)
        if dists.size == 0:
            return False, None, None
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        if dist <= threshold:
            return True, labels[idx], dist
        return False, None, dist
    except Exception as e:
        print('match_persisted_embedding failed', e)
        return False, None, None


def persist_and_update_classifier(emb, label):
    """Append emb/label to persisted embeddings and retrain classifier immediately.

    emb: 1D numpy array
    label: string
    Returns dict with retrain result.
    """
    try:
        embs, labels = _load_persisted_embeddings()
        if embs is None or labels is None or len(labels) == 0:
            embs = np.expand_dims(np.asarray(emb, dtype=float), axis=0)
            labels = [label]
        else:
            embs = np.vstack([embs, np.asarray(emb, dtype=float)])
            labels = list(labels) + [label]
        ok = _save_persisted_embeddings(embs, labels)
        # retrain classifier on saved embeddings
        try:
            n_neighbors = min(3, embs.shape[0])
            obj = train_knn(embs, labels, n_neighbors=n_neighbors)
            # notify clients
            try:
                socketio.emit('retrain_complete', {'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn'}, namespace='/alerts')
            except Exception:
                pass
            return {'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn', 'persisted': ok}
        except Exception as e:
            print('persist_and_update_classifier retrain failed', e)
            return {'success': False, 'error': str(e)}
    except Exception as e:
        print('persist_and_update_classifier failed', e)
        return {'success': False, 'error': str(e)}


def _prune_recent_cache():
    """Remove old entries from RECENT_EMBED_CACHE."""
    try:
        now = time.time()
        # mutate in-place to avoid reassigning global
        i = 0
        while i < len(RECENT_EMBED_CACHE):
            if now - RECENT_EMBED_CACHE[i].get('ts', 0) > RECENT_EMBED_TTL:
                RECENT_EMBED_CACHE.pop(i)
            else:
                i += 1
    except Exception:
        pass


def add_recent_embedding(emb, label):
    """Add an embedding to the short-term cache.

    emb: 1D numpy array
    label: string (name)
    """
    try:
        _prune_recent_cache()
        RECENT_EMBED_CACHE.insert(0, {'emb': np.array(emb, dtype=float), 'label': label, 'ts': time.time()})
        # clamp size to avoid unbounded growth
        if len(RECENT_EMBED_CACHE) > 200:
            RECENT_EMBED_CACHE.pop()
    except Exception:
        pass


@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    """Trigger a background retrain of the classifier using images in the DB.
    Returns immediately with a job accepted response. The retrain runs in a daemon thread.
    """
    try:
        t = threading.Thread(target=retrain_classifier, kwargs={'device':'cpu', 'method':'knn'}, daemon=True)
        t.start()
        return jsonify({'success': True, 'message': 'retrain started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/retrain_sync', methods=['POST'])
def retrain_sync_endpoint():
    """Run retrain synchronously and return result JSON. Useful for UI flows that want
    the newly added face to be available immediately after the call completes.
    """
    try:
        res = retrain_classifier(device='cpu', method='knn')
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# latest frames per camera (BGR numpy arrays) used for snapshot endpoints to avoid opening camera twice
LATEST_FRAMES = {}

# SocketIO for real-time alerts. Prefer eventlet when available; restrict CORS to localhost for slightly
# improved security in development. Disable verbose engineio logging by default.
try:
    import eventlet  # noqa: F401
    _async_mode = 'eventlet'
except Exception:
    _async_mode = 'threading'

_cors = '*'
socketio = SocketIO(app, cors_allowed_origins=_cors, async_mode=_async_mode, logger=False, engineio_logger=False)

# Initialize DB
init_db()

# load persisted tracker settings if present
load_tracker_settings()
# load persisted alerts if present
load_alerts()
# Ensure a classifier is available at startup if faces are already in the DB
def ensure_classifier_ready():
    try:
        from model_utils import load_classifier as _load_cls
        if _load_cls() is not None:
            return
        # no classifier persisted yet; try to build from DB if faces exist
        session = SessionLocal()
        faces = session.query(Face).all()
        session.close()
        if not faces:
            return
        embs = []
        labels = []
        for f in faces:
            try:
                emb = embedding_from_path(f.image_path, device='cpu')
                if emb is None:
                    continue
                embs.append(emb)
                labels.append(f.name)
            except Exception:
                continue
        if embs:
            embs = np.stack(embs, axis=0)
            n_neighbors = min(3, embs.shape[0])
            train_knn(embs, labels, n_neighbors=n_neighbors)
            print('Initialized classifier from existing DB faces (', len(labels), 'faces )')
    except Exception as e:
        print('ensure_classifier_ready failed', e)

ensure_classifier_ready()
# load persisted ack keys from DB so alerts are suppressed across tracker restarts
def load_ack_keys_from_db():
    try:
        session = SessionLocal()
        rows = session.query(AlertAck).all()
        session.close()
        if not rows:
            return
        cutoff = time.time() - _ack_expiry_seconds()
        expired_ids = []
        for a in rows:
            try:
                if not a.track_key:
                    continue
                if float(a.ts or 0) < cutoff:
                    expired_ids.append(a.id)
                    continue
                remember_ack_key(a.track_key, ts=a.ts)
            except Exception:
                continue
        if expired_ids:
            try:
                session2 = SessionLocal()
                session2.query(AlertAck).filter(AlertAck.id.in_(expired_ids)).delete(synchronize_session=False)
                session2.commit()
                session2.close()
            except Exception:
                pass
    except Exception as e:
        print('load_ack_keys_from_db failed', e)

load_ack_keys_from_db()

# Start notification worker thread
start_notification_worker()


# Debug: log connect/disconnect on the alerts namespace to diagnose client connect_error
@socketio.on('connect', namespace='/alerts')
def on_alerts_connect():
    try:
        print(f"Socket.IO /alerts connect: sid={request.sid} addr={request.remote_addr} headers={{}}")
    except Exception:
        print('Socket.IO /alerts connect (info unavailable)')


@socketio.on('disconnect', namespace='/alerts')
def on_alerts_disconnect():
    try:
        print(f"Socket.IO /alerts disconnect: sid={request.sid}")
    except Exception:
        print('Socket.IO /alerts disconnect')

# Simple auth for scaffold (replace with real system)
VALID_USER = {'username': 'admin', 'password': 'password'}

# Helper to capture one frame from a camera
def capture_frame(camera_id=0):
    # Prefer using last frame captured by generator to avoid opening a second VideoCapture
    try:
        lf = LATEST_FRAMES.get(str(camera_id))
        if lf is not None:
            return lf.copy()
    except Exception:
        pass
    cap = open_camera(camera_id)
    if cap is None or not cap.isOpened():
        print(f"capture_frame: camera {camera_id} could not be opened")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"capture_frame: failed to read frame from camera {camera_id}")
        return None
    return frame


def open_camera(camera_id=0, preferred_api=None, timeout=2.0):
    """Try to open the camera using several Windows backends and return an opened VideoCapture or None.
    This helps avoid MSMF grabFrame errors by falling back to DirectShow (CAP_DSHOW) or VFW.
    """
    # If camera_id looks like a digit string, cast to int
    try:
        cam_idx = int(camera_id)
    except Exception:
        cam_idx = camera_id

    # Map preferred_api string to cv2 constant if present
    name_map = {
        'dshow': getattr(cv2, 'CAP_DSHOW', None),
        'msmf': getattr(cv2, 'CAP_MSMF', None),
        'vfw': getattr(cv2, 'CAP_VFW', None),
        'default': None,
        'any': None,
    }

    backends = []
    # If a preferred_api string was provided and maps to a valid value, try it first
    if preferred_api:
        p = str(preferred_api).lower()
        api_val = name_map.get(p, None)
        # If api_val is explicitly None but p was 'default' or 'any', we'll try default
        backends.append(api_val)

    # then fall back to common choices: DirectShow, MSMF, VFW, default
    if getattr(cv2, 'CAP_DSHOW', None) is not None and cv2.CAP_DSHOW not in backends:
        backends.append(cv2.CAP_DSHOW)
    if getattr(cv2, 'CAP_MSMF', None) is not None and cv2.CAP_MSMF not in backends:
        backends.append(cv2.CAP_MSMF)
    if getattr(cv2, 'CAP_VFW', None) is not None and cv2.CAP_VFW not in backends:
        backends.append(cv2.CAP_VFW)
    backends.append(None)

    import datetime
    global LAST_CAMERA_ERROR
    for api in backends:
        try:
            if api is None:
                cap = cv2.VideoCapture(cam_idx)
            else:
                cap = cv2.VideoCapture(cam_idx, api)
            start = time.time()
            # give it a short time to initialize
            while time.time() - start < timeout:
                if cap is not None and cap.isOpened():
                    # try to grab a frame
                    ret, _ = cap.read()
                    if ret:
                        print(f"open_camera: opened camera {cam_idx} with api {api}")
                        LAST_CAMERA_ERROR['msg'] = None
                        LAST_CAMERA_ERROR['ts'] = None
                        return cap
                time.sleep(0.05)
            # cleanup and try next
            try:
                cap.release()
            except Exception:
                pass
        except Exception as e:
            msg = f"open_camera: backend {api} failed: {e}"
            print(msg)
            LAST_CAMERA_ERROR['msg'] = msg
            LAST_CAMERA_ERROR['ts'] = datetime.datetime.now().isoformat()
            try:
                cap.release()
            except Exception:
                pass
    msg = f"open_camera: failed to open camera {cam_idx} with any backend"
    print(msg)
    LAST_CAMERA_ERROR['msg'] = msg
    LAST_CAMERA_ERROR['ts'] = datetime.datetime.now().isoformat()
    return None


@app.route('/')
def root():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == VALID_USER['username'] and password == VALID_USER['password']:
            return redirect(url_for('select_camera'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


@app.route('/select_camera', methods=['GET', 'POST'])
def select_camera():
    # For simplicity, offer common choices
    cameras = [0, 1]
    return render_template('select_camera.html', cameras=cameras)


@app.route('/stream')
def stream():
    cam = request.args.get('camera', 0)
    backend = request.args.get('backend', None)
    device = request.args.get('device') or app.config.get('DEFAULT_DEVICE', 'cpu')
    frame_skip = request.args.get('frame_skip') or app.config.get('DEFAULT_FRAME_SKIP', 1)
    show_ids = request.args.get('show_ids', '1')
    show_ids = False if str(show_ids) in ('0', 'false', 'False') else True
    # load tracker settings if present
    settings = TRACKER_SETTINGS.get(str(cam), {})
    return render_template('stream.html', camera=cam, backend=backend, device=device, frame_skip=frame_skip, show_ids=show_ids, tracker_settings=settings)


@app.route('/test_alert', methods=['GET'])
def test_alert():
    """Emit a test alert to connected clients on the /alerts namespace.
    Useful for debugging desktop notifications without requiring the detector.
    """
    try:
        payload = {
            'label': 'Unknown',
            'image': None
        }
        # Emit to the alerts namespace
        socketio.emit('alert', payload, namespace='/alerts')
        return jsonify({'success': True, 'emitted': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug_unknown_with_image', methods=['POST'])
def debug_unknown_with_image():
    """Emit an 'Unknown' alert with a sample face image pulled from the dataset to exercise
    the unknown popup + Add Face flow without needing a live camera frame.
    Optional form/json param: name (preferred image whose filename contains this substring).
    """
    try:
        data = request.get_json(force=False, silent=True) or request.form.to_dict() or {}
        prefer = (data.get('name') or '').strip().lower()
        img_path = None
        # find candidate image in dataset
        try:
            files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            if prefer:
                for f in files:
                    if prefer in f.lower():
                        img_path = os.path.join(DATASET_DIR, f)
                        break
            if img_path is None and files:
                img_path = os.path.join(DATASET_DIR, files[0])
        except Exception:
            img_path = None
        img_b64 = None
        if img_path and os.path.exists(img_path):
            try:
                import base64 as _b64
                import cv2 as _cv
                im = _cv.imread(img_path)
                if im is not None:
                    retc, buf = _cv.imencode('.jpg', im)
                    if retc:
                        img_b64 = _b64.b64encode(buf.tobytes()).decode('utf-8')
            except Exception:
                img_b64 = None
        payload = {'label': 'Unknown', 'camera': '0'}
        if img_b64:
            payload['image'] = img_b64
        try:
            import datetime as _dt
            ALERTS.insert(0, {'ts': _dt.datetime.now().isoformat(), 'payload': payload})
            if len(ALERTS) > 200:
                ALERTS.pop()
            save_alerts()
        except Exception:
            pass
        socketio.emit('alert', payload, namespace='/alerts')
        # also send a Web Push notification so this endpoint mirrors real unknown alerts
        try:
            subs = _load_push_subscriptions()
            if subs:
                push_payload = {
                    'title': payload.get('label') or 'Unknown face detected',
                    'body': 'Click to review or add face.',
                    'image': payload.get('image'),
                    'label': payload.get('label'),
                    'camera': payload.get('camera'),
                    'idx': 0
                }
                # Queue notification for sequential processing
                queue_notification(subs, push_payload)
        except Exception:
            pass
        return jsonify({'success': True, 'emitted': True, 'with_image': bool(img_b64)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/vapid_public_key', methods=['GET'])
def vapid_public_key():
    """Return VAPID public key (base64url) for client-side Push subscription.
    The server expects a JSON file at `models/vapid.json` with keys `publicKey` and `privateKey`.
    If missing, this endpoint will return an error telling the operator to create keys.
    """
    vapid = load_vapid_keys()
    if not vapid or not vapid.get('publicKey'):
        # attempt to generate keys automatically if cryptography is available
        try:
            vapid = generate_vapid_keys()
            return jsonify({'success': True, 'publicKey': vapid.get('publicKey'), 'generated': True})
        except Exception as e:
            return jsonify({'success': False, 'error': 'VAPID public key not configured. Install `cryptography` and run scripts/generate_vapid.py or create models/vapid.json manually.'}), 404
    return jsonify({'success': True, 'publicKey': vapid.get('publicKey')})


@app.route('/subscribe_push', methods=['POST'])
def subscribe_push():
    try:
        sub = request.get_json(force=True)
        if not sub or not sub.get('endpoint'):
            return jsonify({'success': False, 'error': 'invalid subscription'}), 400
        endpoint = sub.get('endpoint', '')
        if 'notify.windows.com' in endpoint.lower():
            msg = ('This browser registered a Windows Notification Service endpoint, which does not accept VAPID keys. '
                   'Please use Chrome, Edge (Chromium), or Firefox and re-allow notifications.')
            print('[push] rejected subscription from Windows Notification Service endpoint; user must re-subscribe via VAPID-ready browser')
            # Respond 200 so the browser console does not log a failed network request; client can inspect success flag.
            return jsonify({'success': False, 'error': 'wns_not_supported', 'message': msg}), 200
        subs = _load_push_subscriptions()
        # keep unique by endpoint
        endpoints = [s.get('endpoint') for s in subs if s.get('endpoint')]
        if sub.get('endpoint') in endpoints:
            return jsonify({'success': True, 'message': 'already subscribed'})
        subs.append(sub)
        _save_push_subscriptions(subs)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/send_push_test', methods=['POST'])
def send_push_test():
    """Send a test push payload to all stored subscriptions (returns per-subscription status)."""
    try:
        subs = _load_push_subscriptions()
        results = []
        # Use the latest persisted alert image if available to include a face thumbnail
        img_b64 = None
        try:
            if ALERTS and len(ALERTS) > 0:
                latest = ALERTS[-1]
                img_b64 = latest.get('image')
        except Exception:
            img_b64 = None
        payload = {'title': 'Unknown face detected', 'body': 'An unknown face was detected. Click to review.', 'icon': None, 'image': img_b64}
        for s in subs:
            ok = send_web_push(s, payload)
            results.append({'endpoint': s.get('endpoint'), 'ok': ok})
        return jsonify({'success': True, 'results': results, 'payload_preview': {'has_image': bool(img_b64)}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/push_alert/<int:idx>', methods=['POST'])
def push_alert(idx):
    """Send a push notification built from saved ALERTS[idx] to all subscriptions."""
    try:
        if idx < 0 or idx >= len(ALERTS):
            return jsonify({'success': False, 'error': 'alert index out of range'}), 400
        alert = ALERTS[idx]
        payload = {
            'title': alert.get('label') or 'Unknown face',
            'body': 'An alert was recorded. Click to add face.',
            'image': alert.get('image')
        }
        subs = _load_push_subscriptions()
        results = []
        for s in subs:
            ok = send_web_push(s, payload)
            results.append({'endpoint': s.get('endpoint'), 'ok': ok})
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/debug_detect', methods=['GET'])
def debug_detect():
    """Capture a single frame from the current camera (default 0), run MTCNN detection and FaceNet embedding/classification,
    and return JSON with boxes, probs, thumbnail base64, and known/label/distance for each detected face. Useful to verify MTCNN/FaceNet pipeline.
    """
    cam = request.args.get('camera', 0)
    try:
        frame = None
        try:
            lf = LATEST_FRAMES.get(str(cam))
            if lf is not None:
                frame = lf.copy()
        except Exception:
            frame = None
        if frame is None:
            frame = capture_frame(cam)
        if frame is None:
            return jsonify({'success': False, 'error': 'Could not capture frame'})

        # prepare PIL image
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mtcnn = get_mtcnn(device='cpu')
        # detect boxes on full image
        try:
            boxes, probs = mtcnn.detect(pil)
        except Exception as e:
            return jsonify({'success': False, 'error': f'MTCNN detect failed: {e}'})
        faces_out = []
        if boxes is None or len(boxes) == 0:
            return jsonify({'success': True, 'faces': []})
        # For each box, extract face tensor and embedding
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            thumb_b64 = None
            try:
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    retc, buf = cv2.imencode('.jpg', crop)
                    if retc:
                        thumb_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            except Exception:
                thumb_b64 = None
            # get aligned face tensor using mtcnn.extract
            try:
                face_tensors = mtcnn.extract(pil, [(x1, y1, x2, y2)])
                if face_tensors is None or len(face_tensors) == 0:
                    emb = None
                    known = False
                    label = None
                    dist = None
                else:
                    face_t = face_tensors[0]
                    emb = embedding_from_face_tensor(face_t, device='cpu')
                    known, label, dist = is_known(emb, threshold=app.config.get('KNOWN_DISTANCE_THRESHOLD', 0.9))
            except Exception as e:
                emb = None
                known = False
                label = None
                dist = None

            faces_out.append({'box': [x1, y1, x2, y2], 'prob': float(prob) if prob is not None else None, 'thumb': thumb_b64, 'known': bool(known), 'label': label, 'distance': dist})

        return jsonify({'success': True, 'faces': faces_out})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
def gen_frames(camera_id=0, backend=None, device='cpu', frame_skip=1):
    # This generator yields multipart JPEG frames with detection boxes drawn
    try:
        frame_skip = max(1, int(frame_skip))
    except Exception:
        frame_skip = 1
    # Motion gating disabled: no gating state
    cap = open_camera(camera_id, preferred_api=backend)
    import datetime
    global LAST_CAMERA_ERROR
    if cap is None or not cap.isOpened():
        # yield a plain image showing camera error repeatedly, with last error message
        while True:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = 'Camera not available'
            err = LAST_CAMERA_ERROR.get('msg')
            if err:
                msg = err[-60:]  # show last part if too long
            cv2.putText(blank, msg, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
    # init mtcnn once for requested device
    mtcnn = get_mtcnn(device=device)
    frame_counter = 0
    # choose a resize factor for faster detection (tune as needed)
    # use a slightly larger factor by default; if detection fails we will retry on full-size
    resize_factor = 0.8
    try:
        while True:
            ret, frame = cap.read()
            # store latest frame for snapshot endpoints
            try:
                cam_key_f = str(camera_id)
                if ret and frame is not None:
                    LATEST_FRAMES[cam_key_f] = frame.copy()
            except Exception:
                pass
            if not ret or frame is None:
                print('gen_frames: read failed, attempting reopen (exponential backoff)')
                # release current capture if possible
                try:
                    cap.release()
                except Exception:
                    pass

                reopened = False
                start_retry = time.time()
                max_retry_seconds = 30.0  # keep trying for up to 30 seconds
                backoff = 0.2
                max_backoff = 5.0

                while time.time() - start_retry < max_retry_seconds:
                    cap = open_camera(camera_id, preferred_api=backend)
                    if cap is not None and cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            reopened = True
                            print('gen_frames: camera reopened')
                            break
                    # sleep then backoff
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 2)

                if not reopened:
                    # yield an error frame so the browser doesn't freeze; we'll continue trying
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, 'Camera read error - retrying', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    ret2, buffer = cv2.imencode('.jpg', blank)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    # short wait before the next loop iteration will attempt reopen again
                    time.sleep(1.0)
                    continue

            # downscale for detection
            small = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            pil_small = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            # increment frame counter and optionally skip detection to save CPU
            frame_counter += 1
            do_detect = (frame_skip <= 1) or (frame_counter % frame_skip == 0)

            # Motion gating disabled: do_detect remains controlled only by frame_skip cadence
            boxes = None
            probs = None
            if do_detect:
                t0 = time.time()
                boxes, probs = mtcnn.detect(pil_small)
                PERF_STATS['detect_total'] += (time.time() - t0)
                # If no faces found, try a few fallbacks to improve detection on low-quality images
                if (boxes is None or len(boxes) == 0):
                    try:
                        # First retry on the full-size image (no downscale)
                        pil_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        boxes_full, probs_full = mtcnn.detect(pil_full)
                        if boxes_full is not None and len(boxes_full) > 0:
                            boxes, probs = boxes_full, probs_full
                        else:
                            # Try contrast-enhanced image to aid detection
                            from PIL import ImageEnhance
                            enhancer = ImageEnhance.Contrast(pil_full)
                            pil_contrast = enhancer.enhance(1.5)
                            boxes_c, probs_c = mtcnn.detect(pil_contrast)
                            if boxes_c is not None and len(boxes_c) > 0:
                                boxes, probs = boxes_c, probs_c
                    except Exception:
                        # silent fallback: continue with boxes possibly None
                        pass
            PERF_STATS['frames'] += 1

            if boxes is not None:
                # scale boxes back to original frame size and collect for tracking
                boxes_scaled = []
                for box in boxes:
                    x1, y1, x2, y2 = [int(b / resize_factor) for b in box]
                    boxes_scaled.append([x1, y1, x2, y2])

                # get or create tracker for this camera
                cam_key = str(camera_id)
                trk = TRACKERS.get(cam_key)
                if trk is None:
                    # use defaults from settings if present
                    s = TRACKER_SETTINGS.get(cam_key, {})
                    iou_t = float(s.get('iou_threshold', 0.3))
                    max_m = int(s.get('max_missed', 5))
                    cooldown = float(s.get('alert_cooldown', 5.0))
                    trk = SimpleTracker(iou_threshold=iou_t, max_missed=max_m, alert_cooldown=cooldown)
                    TRACKERS[cam_key] = trk
                else:
                    # ensure tracker uses latest settings
                    s = TRACKER_SETTINGS.get(cam_key, {})
                    trk.iou_threshold = float(s.get('iou_threshold', trk.iou_threshold))
                    trk.max_missed = int(s.get('max_missed', trk.max_missed))
                    trk.alert_cooldown = float(s.get('alert_cooldown', trk.alert_cooldown))

                tracked = trk.update(boxes_scaled)

                # iterate detections with associated track info
                for di, (box, prob) in enumerate(zip(boxes_scaled, probs)):
                    x1, y1, x2, y2 = box
                    thumb_b64 = None
                    try:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size != 0:
                            retc, buf = cv2.imencode('.jpg', crop)
                            if retc:
                                thumb_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                    except Exception:
                        thumb_b64 = None

                    # default values
                    known, name, dist = (False, None, None)

                    # get associated track info early so we can honor any per-track overrides
                    tinfo = None
                    if tracked and di < len(tracked):
                        tinfo = tracked[di]
                    track_id = tinfo['id'] if tinfo else None

                    # Always attempt recognition so a face can switch from Unknown -> Known
                    # (Notification dedupe is handled by mark_unknown_alert later)
                    if True:
                        # If this track was explicitly marked as known/added by the operator, honor that immediately
                        track_state = tinfo.get('track') if tinfo else None
                        if track_state and track_state.get('override_known'):
                            # Treat as known immediately (fast-path) so the client won't be repeatedly prompted
                            known = True
                            name = track_state.get('override_known')
                            dist = None
                        else:
                            try:
                                # extract face tensor for embedding
                                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                face_tensor = mtcnn.extract(pil, [(x1, y1, x2, y2)], save_path=None)
                                if face_tensor is not None and len(face_tensor) > 0:
                                    face_t = face_tensor[0]
                                    t1 = time.time()
                                    emb = embedding_from_face_tensor(face_t, device=device)
                                    PERF_STATS['embed_total'] += (time.time() - t1)
                                    # check recent embedding cache first (fast-path) so we suppress re-prompts
                                    try:
                                        _prune_recent_cache()
                                        matched = False
                                        kn_thresh = float(app.config.get('KNOWN_DISTANCE_THRESHOLD', 0.9))
                                        for ce in RECENT_EMBED_CACHE:
                                            try:
                                                d = float(np.linalg.norm(emb - ce['emb']))
                                                if d <= kn_thresh:
                                                    known = True
                                                    name = ce.get('label')
                                                    dist = d
                                                    matched = True
                                                    print(f"[RECOGNITION] Matched from recent cache: {name}, distance: {d:.3f}, threshold: {kn_thresh}")
                                                    break
                                            except Exception:
                                                continue
                                        if not matched:
                                            # fallback to persisted embeddings cache before classifier to avoid false unknowns
                                            persisted_match, persisted_name, persisted_dist = match_persisted_embedding(emb, kn_thresh)
                                            if persisted_match:
                                                known = True
                                                name = persisted_name
                                                dist = persisted_dist
                                                print(f"[RECOGNITION] Matched from persisted embeddings: {name}, distance: {dist:.3f}, threshold: {kn_thresh}")
                                                try:
                                                    add_recent_embedding(emb, persisted_name)
                                                except Exception:
                                                    pass
                                                matched = True
                                        if not matched:
                                            known, name, dist = is_known(emb, threshold=kn_thresh, prob_threshold=app.config.get('KNOWN_PROB_THRESHOLD', 0.5))
                                            if known:
                                                print(f"[RECOGNITION] Matched from classifier: {name}, distance: {dist:.3f}, threshold: {kn_thresh}")
                                            else:
                                                print(f"[RECOGNITION] Unknown face detected. Distance: {dist}, Threshold: {kn_thresh}")
                                    except Exception as e:
                                        print(f"[RECOGNITION] Exception during matching: {e}")
                                        known, name, dist = is_known(emb, threshold=app.config.get('KNOWN_DISTANCE_THRESHOLD', 0.9), prob_threshold=app.config.get('KNOWN_PROB_THRESHOLD', 0.5))
                            except Exception:
                                known, name, dist = (False, None, None)

                    # Check per-camera show_ids setting (defaults to True)
                    cam_s = TRACKER_SETTINGS.get(str(camera_id), {})
                    show_ids_flag = cam_s.get('show_ids', True)

                    # Build two-line labels: top (Known/Unknown), bottom (name or empty)
                    id_part = (f"ID:{track_id} ") if show_ids_flag and track_id is not None else ''
                    top_label = 'Known' if known else 'Unknown'
                    bottom_label = ''
                    if known and name:
                        bottom_label = str(name)
                    else:
                        # when unknown, show id_part as secondary info if present
                        bottom_label = id_part.strip()

                    # choose color: green for known, red for unknown
                    if known:
                        text_color = (255, 255, 255)
                        box_color = (0, 200, 0)
                    else:
                        text_color = (255, 255, 255)
                        box_color = (0, 0, 200)

                    # draw the rectangle with chosen color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # prepare two-line background and text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 1
                    (tw1, th1), _ = cv2.getTextSize(top_label, font, font_scale, font_thickness)
                    (tw2, th2), _ = cv2.getTextSize(bottom_label, font, font_scale, font_thickness)
                    width_bg = max(tw1, tw2)
                    height_bg = th1 + th2 + 8  # padding between lines
                    text_x = x1
                    text_y_top = max(0, y1 - 6 - th2)  # top baseline
                    # background rectangle (expand a bit for padding)
                    cv2.rectangle(frame, (text_x - 2, text_y_top - th1 - 6), (text_x + width_bg + 4, text_y_top + th2 + 4), box_color, -1)
                    # put top label
                    cv2.putText(frame, top_label, (text_x, text_y_top), font, font_scale, (255,255,255), font_thickness)
                    # put bottom label (below top)
                    bottom_y = text_y_top + th1 + 4
                    if bottom_label:
                        cv2.putText(frame, bottom_label, (text_x, bottom_y), font, font_scale, (255,255,255), font_thickness)

                    if known:
                        try:
                            release_alert_lock(camera_id, track_id=track_id, label=name)
                        except Exception:
                            pass
                    else:
                        track_state = tinfo.get('track') if tinfo else None
                        ack_key = None
                        try:
                            if name and str(name).strip() and str(name).strip().lower() not in ('unknown', ''):
                                ack_key = f"{camera_id}::{str(name).strip().lower()}"
                            elif track_id is not None:
                                ack_key = f"{camera_id}::{track_id}"
                        except Exception:
                            ack_key = None
                        if ack_key and _ack_is_active(ack_key):
                            try:
                                release_alert_lock(camera_id, track_id=track_id, label=name)
                            except Exception:
                                pass
                            continue
                        if not mark_unknown_alert(camera_id, track_id=track_id, box=[x1, y1, x2, y2]):
                            continue
                        try:
                            alert_label = name if name else 'Unknown'
                            payload = {'label': alert_label, 'box': [x1, y1, x2, y2], 'track_id': track_id, 'camera': str(camera_id)}
                            try:
                                payload['track_id'] = int(track_id) if track_id is not None else None
                            except Exception:
                                payload['track_id'] = None
                            if not thumb_b64:
                                try:
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size != 0:
                                        retc, buf = cv2.imencode('.jpg', crop)
                                        if retc:
                                            thumb_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                                except Exception:
                                    thumb_b64 = None
                            if thumb_b64:
                                payload['image'] = thumb_b64
                            try:
                                import datetime as _dt
                                ALERTS.insert(0, {'ts': _dt.datetime.now().isoformat(), 'payload': payload})
                                if len(ALERTS) > 200:
                                    ALERTS.pop()
                                save_alerts()
                            except Exception:
                                pass
                            try:
                                payload['idx'] = 0
                            except Exception:
                                pass
                            socketio.emit('alert', payload, namespace='/alerts')
                            try:
                                subs = _load_push_subscriptions()
                                if subs:
                                    push_payload = {
                                        'title': payload.get('label') or 'Unknown face detected',
                                        'body': 'Click to open or add face',
                                        'image': payload.get('image'),
                                        'label': payload.get('label'),
                                        'camera': payload.get('camera'),
                                        'track_id': payload.get('track_id'),
                                        'idx': payload.get('idx', 0)
                                    }
                                    # Queue notification for sequential processing
                                    queue_notification(subs, push_payload)
                            except Exception:
                                pass
                            if track_state:
                                try:
                                    trk.mark_alerted(track_state)
                                except Exception:
                                    pass
                        except Exception:
                            pass

            # encode and yield
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        try:
            cap.release()
        except Exception:
            pass


@app.route('/video_feed')
def video_feed():
    camera_id = request.args.get('camera', 0)
    backend = request.args.get('backend', None)
    device = request.args.get('device', 'cpu')
    frame_skip = int(request.args.get('frame_skip', 1) or 1)
    show_ids = request.args.get('show_ids', '1')
    show_ids = False if str(show_ids) in ('0', 'false', 'False') else True
    return Response(gen_frames(camera_id, backend=backend, device=device, frame_skip=frame_skip), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot', methods=['POST'])
def snapshot():
    camera_id = request.form.get('camera', 0)
    # Prefer the in-memory latest frame captured by the stream generator so snapshot matches displayed boxes
    frame = None
    try:
        lf = LATEST_FRAMES.get(str(camera_id))
        if lf is not None:
            frame = lf.copy()
    except Exception:
        frame = None
    if frame is None:
        frame = capture_frame(camera_id)
    if frame is None:
        return jsonify({'success': False, 'error': 'Could not capture frame'})
    ts = int(time.time())
    filename = f'snap_{camera_id}_{ts}.jpg'
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return jsonify({'success': True, 'path': url_for('static_file', filename=f'snapshots/{filename}')})


@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route('/add_face', methods=['POST'])
def add_face():
    # expects multipart with image or snapshot path, name and profession
    name = request.form.get('name')
    camera_id = request.form.get('camera', 0)
    profession = request.form.get('profession')
    track_id = request.form.get('track_id')
    image_data = request.files.get('image')
    image_path = None
    if image_data:
        ts = int(time.time())
        filename = f'{name.replace(" ", "_")}_{ts}.jpg'
        path = os.path.join(DATASET_DIR, filename)
        image_data.save(path)
        image_path = path
    else:
        return jsonify({'success': False, 'error': 'No image provided'})

    # insert DB
    session = SessionLocal()
    face = Face(name=name, profession=profession, image_path=image_path)
    session.add(face)
    session.commit()
    session.close()
    # If the client provided a track id, mark that track as recently alerted so the same ID
    # won't immediately re-alert while retraining completes. Then run a synchronous retrain
    # so the newly added face is available for recognition immediately.
    try:
        if track_id is not None:
            try:
                cam_key = str(request.form.get('camera', 0))
                trk = TRACKERS.get(cam_key)
                if trk is not None:
                    tid = int(track_id) if str(track_id).isdigit() else None
                    if tid and tid in trk.tracks:
                        # mark recently alerted so the same track won't immediately re-alert
                        trk.tracks[tid]['last_alert'] = time.time()
                        # set an override so the running tracker treats this track as known immediately
                        try:
                            trk.tracks[tid]['override_known'] = name
                        except Exception:
                            pass
                        # attempt to compute an embedding from the saved image and attach to the track (best-effort)
                        try:
                            img = Image.open(image_path).convert('RGB')
                            # try detect_and_align first
                            f_tensors = detect_and_align(img, device='cpu')
                            if not f_tensors:
                                # if the image is a cropped face, extract full-image box
                                w, h = img.size
                                try:
                                    mt = get_mtcnn(device='cpu')
                                    f_tensors = mt.extract(img, [(0, 0, w, h)])
                                except Exception:
                                    f_tensors = []
                            if f_tensors:
                                ft = f_tensors[0]
                                try:
                                    emb_now = embedding_from_face_tensor(ft, device='cpu')
                                    trk.tracks[tid]['embedding'] = emb_now
                                    # add to recent cache so other tracks with different ids match immediately
                                    try:
                                        add_recent_embedding(emb_now, name)
                                    except Exception:
                                        pass
                                    # persist embedding and update classifier immediately so DB-known faces are recognized
                                    try:
                                        persist_and_update_classifier(emb_now, name)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                pass
        # run a blocking retrain: build embeddings from DB images (including the newly added one)
        try:
            session = SessionLocal()
            faces = session.query(Face).all()
            session.close()
            embs = []
            labels = []
            for f in faces:
                try:
                    emb = embedding_from_path(f.image_path, device='cpu')
                    if emb is None:
                        print('add_face: no embedding for', f.image_path)
                        continue
                    embs.append(emb)
                    labels.append(f.name)
                except Exception as e:
                    print('add_face: failed to extract embedding for', f.image_path, e)
                    continue
            if not embs:
                retr_res = {'success': False, 'error': 'no embeddings'}
            else:
                embs = np.stack(embs, axis=0)
                from model_utils import train_knn
                n_neighbors = min(3, embs.shape[0])
                obj = train_knn(embs, labels, n_neighbors=n_neighbors)
                retr_res = {'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn'}
        except Exception as e:
            print('add_face retrain inner failed', e)
            retr_res = {'success': False, 'error': str(e)}
    except Exception as e:
        print('add_face retrain failed', e)
        return jsonify({'success': True, 'retrain': False, 'error': str(e)})
    try:
        # notify connected clients that retrain completed and a face was added
        try:
            socketio.emit('retrain_complete', retr_res, namespace='/alerts')
        except Exception:
            pass
        try:
            # persist an acknowledgement record so operator clients share dedupe state
            try:
                ack_key = None
                cam_str = str(camera_id)
                if track_id:
                    ack_key = f"{cam_str}::{track_id}"
                else:
                    ack_key = f"{cam_str}::{name.lower()}"
                session2 = SessionLocal()
                import time as _t
                a = AlertAck(camera=cam_str, track_key=ack_key, name=name, ts=int(_t.time()))
                session2.add(a)
                session2.commit()
                # update in-memory ack keys
                try:
                    remember_ack_key(ack_key, ts=a.ts)
                except Exception:
                    pass
                try:
                    release_alert_lock(camera_id, track_id=track_id, label=name)
                except Exception:
                    pass
                session2.close()
            except Exception:
                pass
            socketio.emit('face_added', {'name': name, 'camera': str(camera_id), 'track_id': track_id}, namespace='/alerts')
        except Exception:
            pass
        # persist a server-side acknowledgement so other clients won't re-prompt
        try:
            session2 = SessionLocal()
            key = None
            if track_id:
                key = f"{camera_id}::{track_id}"
            else:
                # fallback to name-based key
                key = f"{camera_id}::{name.lower()}"
            ack = AlertAck(camera=str(camera_id), track_key=key, name=name, ts=int(time.time()))
            session2.add(ack)
            session2.commit()
            try:
                remember_ack_key(key, ts=ack.ts)
            except Exception:
                pass
            try:
                release_alert_lock(camera_id, track_id=track_id, label=name)
            except Exception:
                pass
            session2.close()
        except Exception:
            pass
    except Exception:
        pass
    return jsonify({'success': True, 'retrain': retr_res})


@app.route('/upload_face', methods=['POST'])
def upload_face_from_file():
    """Allow admins to upload a still photo from disk, store it as a Face, and retrain."""
    try:
        name = request.form.get('name')
        profession = request.form.get('profession', '')
        image_file = request.files.get('image')
        if not name:
            return jsonify({'success': False, 'error': 'name required'}), 400
        if not image_file or not image_file.filename:
            return jsonify({'success': False, 'error': 'image file required'}), 400
        face_tensor = None
        face_image = None
        try:
            raw = image_file.read()
            if not raw:
                return jsonify({'success': False, 'error': 'empty image upload'}), 400
            with Image.open(io.BytesIO(raw)).convert('RGB') as img:
                face_tensor = _extract_face_tensor(img, device='cpu')
            if face_tensor is not None:
                emb = embedding_from_face_tensor(face_tensor, device='cpu')
                face_image = _face_tensor_to_image(face_tensor)
            else:
                emb = None
        except Exception as e:
            print('upload_face: failed to parse image', e)
            emb = None
        if emb is None:
            return jsonify({'success': False, 'error': 'could not extract face embedding'}), 400
        try:
            add_recent_embedding(emb, name)
        except Exception:
            pass

        dataset_path = None
        if face_image is not None:
            try:
                safe = ''.join(ch for ch in name if ch.isalnum() or ch in ('_', '-')) or 'face'
                ts = int(time.time())
                filename = f'{safe}_{ts}.jpg'
                dataset_path = os.path.join(DATASET_DIR, filename)
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                face_image.save(dataset_path, format='JPEG')
            except Exception as e:
                print('upload_face: failed to persist dataset image', e)
                dataset_path = None

        try:
            emb_path = save_embedding_vector(emb, name)
        except Exception as e:
            return jsonify({'success': False, 'error': f'failed to persist embedding: {e}'}), 500

        session = SessionLocal()
        face = Face(name=name, profession=profession, image_path=emb_path)
        session.add(face)
        session.commit()
        session.close()

        retrain_info = None
        retrain_info = persist_and_update_classifier(emb, name)
        if not retrain_info or not retrain_info.get('success'):
            retrain_info = retrain_classifier(device='cpu', method='knn')

        try:
            socketio.emit('face_added', {'name': name, 'camera': 'upload', 'track_id': None}, namespace='/alerts')
        except Exception:
            pass

        try:
            session2 = SessionLocal()
            ack_key = f"upload::{name.lower()}"
            a = AlertAck(camera='upload', track_key=ack_key, name=name, ts=int(time.time()))
            session2.add(a)
            session2.commit()
            try:
                remember_ack_key(ack_key, ts=a.ts)
            except Exception:
                pass
            session2.close()
        except Exception:
            pass

        return jsonify({'success': True, 'retrain': retrain_info, 'face': {'name': name, 'image_path': emb_path, 'dataset_image': dataset_path}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/snapshot_faces', methods=['POST'])
def snapshot_faces():
    """Capture a frame, detect faces, return thumbnails and known/unknown flags for each face."""
    camera_id = request.form.get('camera', 0)
    # prefer the latest in-memory frame captured by the stream generator when available
    frame = None
    try:
        lf = LATEST_FRAMES.get(str(camera_id))
        if lf is not None:
            frame = lf.copy()
    except Exception:
        frame = None
    if frame is None:
        frame = capture_frame(camera_id)
    if frame is None:
        return jsonify({'success': False, 'error': 'Could not capture frame'})
    device = 'cpu'
    mtcnn = get_mtcnn(device=device)
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(pil)
    faces = []
    if boxes is None:
        return jsonify({'success': True, 'faces': []})
    for box, prob in zip(boxes, probs):
        try:
            x1, y1, x2, y2 = [int(b) for b in box]
            # crop and encode thumbnail
            crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            thumb_b64 = None
            if crop.size != 0:
                retc, buf = cv2.imencode('.jpg', crop)
                if retc:
                    thumb_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            # extract embedding if possible
            face_tensor = mtcnn.extract(pil, [(x1, y1, x2, y2)])
            known = False
            name = None
            dist = None
            if face_tensor is not None and len(face_tensor) > 0:
                emb = embedding_from_face_tensor(face_tensor[0], device=device)
                known, name, dist = is_known(emb, threshold=app.config.get('KNOWN_DISTANCE_THRESHOLD', 0.9))
            faces.append({'bbox': [x1, y1, x2, y2], 'prob': float(prob) if prob is not None else None, 'known': bool(known), 'name': name, 'distance': dist, 'image': thumb_b64})
        except Exception:
            continue
    return jsonify({'success': True, 'faces': faces})


@app.route('/add_faces_bulk', methods=['POST'])
def add_faces_bulk():
    """Accept multiple uploaded images under a single name+profession and save to dataset + DB."""
    name = request.form.get('name')
    profession = request.form.get('profession')
    files = request.files.getlist('images')
    if not name or not files:
        return jsonify({'success': False, 'error': 'Name and images are required'})
    saved = 0
    session = SessionLocal()
    for f in files:
        if f and f.filename:
            ts = int(time.time())
            safe = name.replace(' ', '_')
            filename = f'{safe}_{ts}_{saved}.jpg'
            path = os.path.join(DATASET_DIR, filename)
            try:
                f.save(path)
                face = Face(name=name, profession=profession, image_path=path)
                session.add(face)
                saved += 1
            except Exception as e:
                print('Failed to save', e)
                continue
    session.commit()
    session.close()
    return jsonify({'success': True, 'saved': saved})


@app.route('/add_face_from_alert', methods=['POST'])
def add_face_from_alert():
    """Create a Face entry from a stored ALERTS entry (thumbnail image) and retrain.
    POST params (form or JSON): alert_idx (int, required), name (required), profession (optional)
    Returns retrain result and created Face info.
    """
    try:
        data = request.get_json(force=False, silent=True) or request.form.to_dict() or {}
        idx = data.get('alert_idx')
        name = data.get('name')
        profession = data.get('profession') or ''
        if idx is None:
            return jsonify({'success': False, 'error': 'alert_idx required'}), 400
        try:
            idx = int(idx)
        except Exception:
            return jsonify({'success': False, 'error': 'invalid alert_idx'}), 400
        if idx < 0 or idx >= len(ALERTS):
            return jsonify({'success': False, 'error': 'alert_idx out of range'}), 400
        if not name:
            return jsonify({'success': False, 'error': 'name required'}), 400

        payload = ALERTS[idx].get('payload', {})
        img_b64 = payload.get('image')
        if not img_b64:
            return jsonify({'success': False, 'error': 'alert has no image'}), 400

        # save image
        try:
            import base64 as _b64
            ts = int(time.time())
            safe = str(name).replace(' ', '_')
            filename = f'{safe}_{ts}_alert.jpg'
            path = os.path.join(DATASET_DIR, filename)
            with open(path, 'wb') as f:
                f.write(_b64.b64decode(img_b64))
        except Exception as e:
            return jsonify({'success': False, 'error': f'failed to save image: {e}'}), 500

        # insert DB Face
        try:
            session = SessionLocal()
            face = Face(name=name, profession=profession, image_path=path)
            session.add(face)
            session.commit()
            session.close()
        except Exception as e:
            return jsonify({'success': False, 'error': f'failed to insert Face: {e}'}), 500

        # compute embedding for this face and add to recent cache + persisted embeddings
        try:
            img = Image.open(path).convert('RGB')
            f_tensors = detect_and_align(img, device='cpu')
            if not f_tensors:
                try:
                    w, h = img.size
                    mt = get_mtcnn(device='cpu')
                    f_tensors = mt.extract(img, [(0, 0, w, h)])
                except Exception:
                    f_tensors = []
            if f_tensors:
                ft = f_tensors[0]
                emb = embedding_from_face_tensor(ft, device='cpu')
                try:
                    add_recent_embedding(emb, name)
                except Exception:
                    pass
                try:
                    persist_and_update_classifier(emb, name)
                except Exception:
                    pass
        except Exception:
            pass

        # run a blocking retrain to ensure classifier is updated for all DB images
        try:
            retr_res = retrain_classifier(device='cpu', method='knn')
        except Exception as e:
            retr_res = {'success': False, 'error': str(e)}

        # create server-side ack so UI dedupe picks this up
        try:
            ack_key = f"{payload.get('camera','0')}::{name.lower()}"
            session2 = SessionLocal()
            import time as _t
            a = AlertAck(camera=str(payload.get('camera','0')), track_key=ack_key, name=name, ts=int(_t.time()))
            session2.add(a)
            session2.commit()
            try:
                remember_ack_key(ack_key, ts=a.ts)
            except Exception:
                pass
            try:
                release_alert_lock(payload.get('camera', '0'), label=name)
            except Exception:
                pass
            session2.close()
        except Exception:
            pass

        return jsonify({'success': True, 'retrain': retr_res, 'face': {'name': name, 'profession': profession, 'image_path': path}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    # Walk dataset, compute embeddings and labels
    device = 'cpu'
    session = SessionLocal()
    faces = session.query(Face).all()
    session.close()
    if not faces:
        return jsonify({'success': False, 'error': 'No faces to train on'})
    embs = []
    labels = []
    for f in faces:
        emb = embedding_from_path(f.image_path, device=device)
        if emb is None:
            continue
        embs.append(emb)
        labels.append(f.name)
    if not embs:
        return jsonify({'success': False, 'error': 'No embeddings extracted'})
    embs = np.stack(embs, axis=0)
    # allow method selection via form param
    method = request.form.get('method', 'knn')
    if method == 'knn':
        obj = train_knn(embs, labels, n_neighbors=3)
    else:
        # svm
        from model_utils import train_classifier
        obj = train_classifier(embs, labels, method='svm', svm_C=1.0)
    return jsonify({'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn'})


@app.route('/debug_alert', methods=['POST'])
def debug_alert():
    """Admin endpoint to trigger an alert. Accepts optional 'camera' and 'label'."""
    camera_id = request.form.get('camera', 0)
    label = request.form.get('label', 'Manual Alert')
    # capture a frame and try to extract first face thumbnail
    frame = capture_frame(camera_id)
    thumb_b64 = None
    box = None
    if frame is not None:
        try:
            from facenet_pytorch import MTCNN
            mt = MTCNN(keep_all=True, device='cpu')
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs = mt.detect(pil)
            if boxes is not None and len(boxes) > 0:
                b = boxes[0]
                x1, y1, x2, y2 = [int(v) for v in b]
                box = [x1, y1, x2, y2]
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    retc, buf = cv2.imencode('.jpg', crop)
                    if retc:
                        thumb_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        except Exception:
            pass
    payload = {'label': label}
    if box:
        payload['box'] = box
    if thumb_b64:
        payload['image'] = thumb_b64
    try:
        # store copy of debug alert for admin UI
        try:
            import datetime as _dt
            ALERTS.insert(0, {'ts': _dt.datetime.now().isoformat(), 'payload': payload})
            if len(ALERTS) > 200:
                ALERTS.pop()
            save_alerts()
        except Exception:
            pass
        socketio.emit('alert', payload, namespace='/alerts')
    except Exception:
        pass
    return jsonify({'success': True, 'payload': {'box': box is not None, 'image': thumb_b64 is not None}})


@app.route('/api/dedupe', methods=['GET'])
def api_dedupe_list():
    try:
        session = SessionLocal()
        cutoff = time.time() - _ack_expiry_seconds()
        rows = (session.query(AlertAck)
                .filter(AlertAck.ts >= cutoff)
                .order_by(AlertAck.ts.desc())
                .limit(500)
                .all())
        session.close()
        acks = [{'id': r.id, 'camera': r.camera, 'track_key': r.track_key, 'name': r.name, 'ts': r.ts} for r in rows]
        return jsonify({'success': True, 'acks': acks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/dedupe/clear', methods=['POST'])
def api_dedupe_clear():
    try:
        data = request.get_json(force=True)
    except Exception:
        data = request.form.to_dict() or {}
    try:
        session = SessionLocal()
        if data.get('clear_all'):
            session.query(AlertAck).delete()
            session.commit()
            session.close()
            ACK_KEYS.clear()
            return jsonify({'success': True, 'cleared': 'all'})
        if data.get('id'):
            try:
                iid = int(data.get('id'))
                row = session.query(AlertAck).filter(AlertAck.id == iid).one_or_none()
                if row and row.track_key:
                    clear_ack_key(row.track_key)
                session.query(AlertAck).filter(AlertAck.id == iid).delete()
                session.commit()
                session.close()
                return jsonify({'success': True, 'cleared': iid})
            except Exception as e:
                session.close()
                return jsonify({'success': False, 'error': str(e)})
        session.close()
        return jsonify({'success': False, 'error': 'no id or clear_all provided'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/admin/dedupe')
def admin_dedupe_page():
    return render_template('admin_dedupe.html')


@app.route('/ack_alert', methods=['POST'])
def ack_alert():
    """Client can acknowledge an alert without adding a face. Expects JSON or form with camera and track_id or name.

    Creates an AlertAck entry and emits face_added to clients so dedupe updates.
    """
    try:
        data = request.get_json(force=False, silent=True) or request.form.to_dict() or {}
        camera = str(data.get('camera', data.get('cam', 0)))
        track_id = data.get('track_id') or data.get('track')
        name = data.get('name') or data.get('label')
        # compute track_key similar to add_face
        if track_id:
            track_key = f"{camera}::{track_id}"
        elif name:
            track_key = f"{camera}::{str(name).lower()}"
        else:
            return jsonify({'success': False, 'error': 'need track_id or name/caption'}), 400

        session = SessionLocal()
        import time as _t
        a = AlertAck(camera=camera, track_key=track_key, name=name, ts=int(_t.time()))
        session.add(a)
        session.commit()
        aid = a.id
        try:
            remember_ack_key(track_key, ts=a.ts)
        except Exception:
            pass
        try:
            release_alert_lock(camera, track_id=track_id, label=name)
        except Exception:
            pass
        session.close()

        # If an image file or base64 image is provided along with name/profession, save as a Face and retrain
        # support multipart 'image' file or JSON 'image_b64'
        image_path = None
        prof = None
        try:
            prof = data.get('profession') or request.form.get('profession')
        except Exception:
            prof = None
        try:
            img_file = request.files.get('image') if hasattr(request, 'files') else None
            if img_file and img_file.filename:
                ts2 = int(time.time())
                safe = (str(name) if name else 'face').replace(' ', '_')
                fname = f"{safe}_{ts2}.jpg"
                path = os.path.join(DATASET_DIR, fname)
                img_file.save(path)
                image_path = path
        except Exception:
            image_path = None
        # base64 image support
        if not image_path:
            try:
                img_b64 = data.get('image_b64')
                if img_b64:
                    import base64
                    ts2 = int(time.time())
                    safe = (str(name) if name else 'face').replace(' ', '_')
                    fname = f"{safe}_{ts2}.jpg"
                    path = os.path.join(DATASET_DIR, fname)
                    with open(path, 'wb') as f:
                        f.write(base64.b64decode(img_b64))
                    image_path = path
            except Exception:
                image_path = None

        # If provided an image and name, add to Face DB and retrain immediately
        retr_res = None
        if image_path and name:
            try:
                session2 = SessionLocal()
                face = Face(name=name, profession=prof or '', image_path=image_path)
                session2.add(face)
                session2.commit()
                session2.close()
            except Exception as e:
                print('ack_alert: failed to add Face row', e)

            # trigger retrain similar to /add_face
            try:
                session3 = SessionLocal()
                faces = session3.query(Face).all()
                session3.close()
                embs = []
                labels = []
                for f in faces:
                    try:
                        emb = embedding_from_path(f.image_path, device='cpu')
                        if emb is None:
                            print('ack_alert: no embedding for', f.image_path)
                            continue
                        embs.append(emb)
                        labels.append(f.name)
                    except Exception as e:
                        print('ack_alert: failed to extract embedding for', f.image_path, e)
                        continue
                if not embs:
                    retr_res = {'success': False, 'error': 'no embeddings'}
                else:
                    embs = np.stack(embs, axis=0)
                    from model_utils import train_knn
                    n_neighbors = min(3, embs.shape[0])
                    obj = train_knn(embs, labels, n_neighbors=n_neighbors)
                    retr_res = {'success': True, 'method': obj.get('method') if isinstance(obj, dict) else 'knn'}
            except Exception as e:
                print('ack_alert retrain failed', e)
                retr_res = {'success': False, 'error': str(e)}

        # notify clients
        try:
            socketio.emit('face_added', {'name': name, 'camera': camera, 'track_id': track_id}, namespace='/alerts')
        except Exception:
            pass
        out = {'success': True, 'id': aid, 'track_key': track_key}
        if retr_res is not None:
            out['retrain'] = retr_res
        return jsonify(out)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/calibrate', methods=['GET', 'POST'])
def calibrate():
    """GET returns current thresholds and classifier method. POST with JSON or form-data to set thresholds.
    POST params: mode=('knn'|'svm'), threshold (float). For knn threshold refers to distance (smaller better).
    For svm threshold refers to probability (0..1, higher better).
    """
    if request.method == 'GET':
        cls = load_classifier()
        method = None
        if isinstance(cls, dict):
            method = cls.get('method')
        return jsonify({'KNOWN_DISTANCE_THRESHOLD': app.config.get('KNOWN_DISTANCE_THRESHOLD'), 'KNOWN_PROB_THRESHOLD': app.config.get('KNOWN_PROB_THRESHOLD', 0.5), 'classifier_method': method})
    # POST
    mode = request.form.get('mode') or request.json.get('mode') if request.json else None
    thr = request.form.get('threshold') or (request.json.get('threshold') if request.json else None)
    if mode is None or thr is None:
        return jsonify({'success': False, 'error': 'mode and threshold required'}), 400
    try:
        thr = float(thr)
    except Exception:
        return jsonify({'success': False, 'error': 'invalid threshold value'}), 400
    mode = mode.lower()
    if mode == 'knn':
        app.config['KNOWN_DISTANCE_THRESHOLD'] = thr
    elif mode == 'svm':
        app.config['KNOWN_PROB_THRESHOLD'] = thr
    else:
        return jsonify({'success': False, 'error': 'unknown mode'}), 400
    return jsonify({'success': True, 'mode': mode, 'threshold': thr})


@app.route('/tracker_config', methods=['GET', 'POST'])
def tracker_config():
    """Get or set tracker configuration for a camera. Uses form-data or JSON.
    GET params: camera
    POST form: camera, iou_threshold, max_missed, alert_cooldown, show_ids
    """
    if request.method == 'GET':
        cam = str(request.args.get('camera', 0))
        s = TRACKER_SETTINGS.get(cam, {})
        return jsonify({'success': True, 'camera': cam, 'settings': s})

    # POST: update settings
    cam = str(request.form.get('camera') or (request.json.get('camera') if request.json else '0'))
    if not cam:
        return jsonify({'success': False, 'error': 'camera required'}), 400
    iou_t = request.form.get('iou_threshold') or (request.json.get('iou_threshold') if request.json else None)
    max_m = request.form.get('max_missed') or (request.json.get('max_missed') if request.json else None)
    cooldown = request.form.get('alert_cooldown') or (request.json.get('alert_cooldown') if request.json else None)
    show_ids = request.form.get('show_ids') or (request.json.get('show_ids') if request.json else None)
    s = TRACKER_SETTINGS.get(cam, {})
    if iou_t is not None:
        try:
            s['iou_threshold'] = float(iou_t)
        except Exception:
            return jsonify({'success': False, 'error': 'invalid iou_threshold'}), 400
    if max_m is not None:
        try:
            s['max_missed'] = int(max_m)
        except Exception:
            return jsonify({'success': False, 'error': 'invalid max_missed'}), 400
    if cooldown is not None:
        try:
            s['alert_cooldown'] = float(cooldown)
        except Exception:
            return jsonify({'success': False, 'error': 'invalid alert_cooldown'}), 400
    if show_ids is not None:
        s['show_ids'] = False if str(show_ids) in ('0', 'false', 'False') else True

    TRACKER_SETTINGS[cam] = s
    # also update existing tracker if present
    trk = TRACKERS.get(cam)
    if trk is not None:
        trk.iou_threshold = float(s.get('iou_threshold', trk.iou_threshold))
        trk.max_missed = int(s.get('max_missed', trk.max_missed))
        trk.alert_cooldown = float(s.get('alert_cooldown', trk.alert_cooldown))
    # persist settings to disk so changes survive restarts
    try:
        save_tracker_settings()
    except Exception:
        pass
    return jsonify({'success': True, 'camera': cam, 'settings': s})


@app.route('/perf_stats')
def perf_stats():
    # return simple averaged times (seconds)
    frames = PERF_STATS.get('frames', 0) or 0
    detect_avg = (PERF_STATS['detect_total'] / frames) if frames else None
    embed_avg = (PERF_STATS['embed_total'] / frames) if frames else None
    return jsonify({'frames': frames, 'detect_avg': detect_avg, 'embed_avg': embed_avg})


@app.route('/admin')
def admin():
    """Admin dashboard: show classifier metadata and tracker settings."""
    # load classifier metadata
    cls = load_classifier()
    classifier_info = None
    if cls is None:
        classifier_info = {'present': False}
    else:
        # cls may be a dict with 'method' and 'clf'
        if isinstance(cls, dict):
            clf = cls.get('clf')
            method = cls.get('method')
        else:
            clf = cls
            method = getattr(clf, 'method', 'knn')
        info = {'present': True, 'method': method}
        try:
            # try to list classes if classifier exposes them
            classes = getattr(clf, 'classes_', None)
            if classes is not None:
                info['classes'] = [str(c) for c in list(classes)]
        except Exception:
            pass
        classifier_info = info

    # show tracker settings (string keys)
    tracker_copy = {str(k): v for k, v in TRACKER_SETTINGS.items()} if TRACKER_SETTINGS else {}

    # find saved classifier file if present
    saved_file = None
    try:
        p = os.path.join(MODEL_DIR, 'classifier.joblib')
        if os.path.exists(p):
            saved_file = p
    except Exception:
        saved_file = None

    return render_template('admin.html', classifier=classifier_info, tracker_settings=tracker_copy, classifier_file=saved_file)


@app.route('/alerts_admin')
def alerts_admin():
    # show recent alerts to the admin
    # provide most recent first
    return render_template('alerts_admin.html', alerts=ALERTS)


@app.route('/alerts/mark_processed', methods=['POST'])
def alerts_mark_processed():
    ts = request.form.get('ts') or (request.json.get('ts') if request.json else None)
    if not ts:
        return jsonify({'success': False, 'error': 'ts required'}), 400
    changed = False
    for a in ALERTS:
        if str(a.get('ts')) == str(ts):
            a['processed'] = True
            changed = True
            break
    if changed:
        save_alerts()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'not found'}), 404


@app.route('/alerts/remove', methods=['POST'])
def alerts_remove():
    ts = request.form.get('ts') or (request.json.get('ts') if request.json else None)
    if not ts:
        return jsonify({'success': False, 'error': 'ts required'}), 400
    idx = None
    for i, a in enumerate(ALERTS):
        if str(a.get('ts')) == str(ts):
            idx = i
            break
    if idx is not None:
        ALERTS.pop(idx)
        save_alerts()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'not found'}), 404


@app.route('/admin/dedupe')
def admin_dedupe():
    # This route was replaced by the unified admin page at /admin/dedupe (see other definitions).
    return render_template('admin_dedupe.html')


@app.route('/debug_last_frame')
def debug_last_frame():
    """Return the most recent frame for a camera as a JPEG so developers can inspect what detection sees."""
    cam = str(request.args.get('camera', 0))
    frame = None
    try:
        frame = LATEST_FRAMES.get(cam)
    except Exception:
        frame = None
    if frame is None:
        # fallback: try to capture a fresh frame
        f = capture_frame(cam)
        if f is None:
            return jsonify({'success': False, 'error': 'No frame available'}), 404
        frame = f
    # encode to JPEG and return bytes
    try:
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({'success': False, 'error': 'Could not encode frame'}), 500
        return Response(buf.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Export tracker settings as JSON
@app.route('/export_tracker_settings')
def export_tracker_settings():
    from flask import jsonify
    return jsonify(TRACKER_SETTINGS)

# Import tracker settings from JSON (POST)
@app.route('/import_tracker_settings', methods=['POST'])
def import_tracker_settings():
    import json
    global TRACKER_SETTINGS
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
        TRACKER_SETTINGS = {str(k): v for k, v in data.items()}
        save_tracker_settings()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# Admin endpoint to fetch last camera error
@app.route('/camera_error')
def camera_error():
    from flask import jsonify
    global LAST_CAMERA_ERROR
    return jsonify({'error': LAST_CAMERA_ERROR.get('msg'), 'timestamp': LAST_CAMERA_ERROR.get('ts')})


@app.route('/refresh_tracker_ids', methods=['POST'])
def refresh_tracker_ids():
    """Reset all tracker IDs to start from 1 for each camera."""
    global TRACKERS
    from tracker import SimpleTracker
    for cam_key in list(TRACKERS.keys()):
        trk = TRACKERS[cam_key]
        # re-initialize tracker for this camera with same settings
        iou_t = getattr(trk, 'iou_threshold', 0.3)
        max_m = getattr(trk, 'max_missed', 5)
        cooldown = getattr(trk, 'alert_cooldown', 5.0)
        TRACKERS[cam_key] = SimpleTracker(iou_threshold=iou_t, max_missed=max_m, alert_cooldown=cooldown)
    return jsonify({'success': True, 'message': 'All tracker IDs reset to start from 1.'})


if __name__ == '__main__':
    # Run with SocketIO (eventlet)
    # For automated test runs we disable the reloader so the process stays in the same PID
    # Allow unsafe werkzeug in local/dev environments to avoid RuntimeError when running with Flask-SocketIO
    certfile = os.environ.get('SSL_CERT')
    keyfile = os.environ.get('SSL_KEY')
    run_kwargs = {'host': '0.0.0.0', 'port': 5000, 'debug': False, 'allow_unsafe_werkzeug': True}
    if certfile and keyfile and os.path.exists(certfile) and os.path.exists(keyfile):
        # Eventlet/gevent accept certfile/keyfile for SSL; this enables HTTPS for LAN access (required for SW/Push)
        run_kwargs.update({'certfile': certfile, 'keyfile': keyfile})
        print(f"Starting HTTPS server with cert: {certfile}")
    socketio.run(app, **run_kwargs)
