import os
import sys
import json
import shutil
import argparse
from collections import defaultdict
from PIL import Image, ImageOps
import numpy as np

# local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database import SessionLocal, Face  # type: ignore
from model_utils import get_mtcnn, detect_and_align, embedding_from_face_tensor, train_knn  # type: ignore

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CLEAN_DIR = os.path.join(BASE_DIR, 'dataset_clean')
BAD_DIR = os.path.join(BASE_DIR, 'dataset_bad')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_PATH = os.path.join(MODEL_DIR, 'hygiene_report.json')


def safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in (name or '').strip())


def load_faces():
    session = SessionLocal()
    rows = session.query(Face).all()
    session.close()
    return rows


def ensure_dirs():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(BAD_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def align_primary_face(img: Image.Image, device='cpu'):
    mt = get_mtcnn(device=device)
    # detect boxes on the full image
    boxes, probs = mt.detect(img)
    if boxes is None or len(boxes) == 0:
        return None, None, None
    # choose largest box
    areas = [(int(b[2]-b[0]) * int(b[3]-b[1])) for b in boxes]
    idx = int(np.argmax(areas))
    box = boxes[idx]
    prob = probs[idx] if probs is not None else None
    # extract aligned face
    x1, y1, x2, y2 = [int(v) for v in box]
    faces = mt.extract(img, [(x1, y1, x2, y2)])
    if faces is None or len(faces) == 0:
        return None, None, None
    return faces[0], (x1, y1, x2, y2), prob


def compute_embedding(face_tensor):
    return embedding_from_face_tensor(face_tensor, device='cpu')


def main():
    ap = argparse.ArgumentParser(description='Dataset hygiene: validate, align, deduplicate, and optionally apply changes')
    ap.add_argument('--apply', action='store_true', help='Apply changes (otherwise dry-run)')
    ap.add_argument('--min-samples', type=int, default=2, help='Minimum samples per person to keep')
    ap.add_argument('--dedup-threshold', type=float, default=0.6, help='Euclidean distance threshold to consider images of same person duplicates (smaller is closer)')
    ap.add_argument('--drop-insufficient', action='store_true', help='Drop persons with fewer than min-samples')
    ap.add_argument('--retrain', action='store_true', help='Retrain classifier after applying changes')
    args = ap.parse_args()

    ensure_dirs()

    report = {
        'checked': 0,
        'missing_files': [],
        'no_face': [],
        'aligned_saved': 0,
        'duplicates': [],
        'insufficient_samples': {},
        'applied': bool(args.apply),
    }

    rows = load_faces()
    name_groups = defaultdict(list)
    emb_index = defaultdict(list)  # name -> list of (id, emb, path)

    for r in rows:
        report['checked'] += 1
        name = (r.name or '').strip()
        sname = safe_name(name or 'unknown')
        path = r.image_path
        if not path or not os.path.exists(path):
            report['missing_files'].append({'id': r.id, 'name': name, 'path': path})
            if args.apply:
                # remove row
                session = SessionLocal()
                try:
                    session.query(Face).filter(Face.id == r.id).delete()
                    session.commit()
                finally:
                    session.close()
            continue
        try:
            img = Image.open(path)
            try:
                # fix EXIF orientation
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img = img.convert('RGB')
        except Exception:
            report['no_face'].append({'id': r.id, 'name': name, 'path': path, 'reason': 'unreadable'})
            if args.apply:
                # move to BAD and drop from DB
                try:
                    shutil.move(path, os.path.join(BAD_DIR, os.path.basename(path)))
                except Exception:
                    pass
                session = SessionLocal()
                try:
                    session.query(Face).filter(Face.id == r.id).delete()
                    session.commit()
                finally:
                    session.close()
            continue

        # align face
        face_t, box, prob = align_primary_face(img, device='cpu')
        if face_t is None:
            report['no_face'].append({'id': r.id, 'name': name, 'path': path, 'reason': 'no-detection'})
            if args.apply:
                try:
                    shutil.move(path, os.path.join(BAD_DIR, os.path.basename(path)))
                except Exception:
                    pass
                session = SessionLocal()
                try:
                    session.query(Face).filter(Face.id == r.id).delete()
                    session.commit()
                finally:
                    session.close()
            continue

        # save aligned image
        try:
            aligned_img = Image.fromarray((np.clip(face_t.permute(1,2,0).cpu().numpy()*255, 0, 255)).astype('uint8'))
            out_name = f"{sname}_{r.id}.jpg"
            out_path = os.path.join(CLEAN_DIR, out_name)
            aligned_img.save(out_path, format='JPEG', quality=92, optimize=True)
            report['aligned_saved'] += 1
            if args.apply:
                # update DB path
                session = SessionLocal()
                try:
                    obj = session.query(Face).get(r.id)
                    if obj:
                        obj.image_path = out_path
                        obj.name = name.strip()
                        session.commit()
                finally:
                    session.close()
        except Exception:
            # if save failed, keep original path
            pass

        # compute embedding for dedup
        try:
            emb = compute_embedding(face_t)
            emb_index[name].append({'id': r.id, 'emb': emb, 'path': out_path if args.apply else path})
        except Exception:
            pass
        name_groups[name].append(r.id)

    # deduplicate within each name
    for name, items in emb_index.items():
        n = len(items)
        to_remove = set()
        for i in range(n):
            if items[i]['id'] in to_remove:
                continue
            for j in range(i+1, n):
                if items[j]['id'] in to_remove:
                    continue
                try:
                    d = float(np.linalg.norm(items[i]['emb'] - items[j]['emb']))
                    if d <= args.dedup_threshold:
                        # mark j as duplicate of i
                        to_remove.add(items[j]['id'])
                        report['duplicates'].append({'keep': items[i]['id'], 'drop': items[j]['id'], 'name': name, 'distance': d})
                except Exception:
                    continue
        if args.apply and to_remove:
            session = SessionLocal()
            try:
                for rid in to_remove:
                    # remove file if it's in CLEAN_DIR
                    try:
                        row = next((x for x in items if x['id'] == rid), None)
                        if row and row['path'] and os.path.exists(row['path']):
                            os.remove(row['path'])
                    except Exception:
                        pass
                    session.query(Face).filter(Face.id == rid).delete()
                session.commit()
            finally:
                session.close()

    # min samples enforcement
    for name, ids in name_groups.items():
        if len(ids) < args.min_samples:
            report['insufficient_samples'][name] = len(ids)
            if args.apply and args.drop_insufficient:
                session = SessionLocal()
                try:
                    for rid in ids:
                        session.query(Face).filter(Face.id == rid).delete()
                    session.commit()
                finally:
                    session.close()

    # save report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    # retrain if requested and applied
    if args.apply and args.retrain:
        # rebuild embeddings from remaining DB rows and train KNN
        session = SessionLocal()
        rows2 = session.query(Face).all()
        session.close()
        if rows2:
            embs = []
            labels = []
            mt = get_mtcnn(device='cpu')
            for r in rows2:
                try:
                    img = Image.open(r.image_path).convert('RGB')
                    faces = mt(img)
                    if faces is None:
                        continue
                    emb = embedding_from_face_tensor(faces, device='cpu')
                    embs.append(emb)
                    labels.append(r.name)
                except Exception:
                    continue
            if embs:
                embs = np.stack(embs, axis=0)
                train_knn(embs, labels, n_neighbors=min(3, len(labels)))
                print('Retrain complete.')


if __name__ == '__main__':
    main()
