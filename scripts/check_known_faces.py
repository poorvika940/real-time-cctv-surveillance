"""Utility script to verify DB faces are recognized as known.
Run via: .\\.venv\\Scripts\\python.exe scripts\\check_known_faces.py
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

from PIL import Image
import torch
import numpy as np

# Ensure project root is importable when script is executed from /scripts
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database import SessionLocal, Face
from model_utils import (
    detect_and_align,
    embedding_from_face_tensor,
    get_mtcnn,
    is_known,
)

KNOWN_DISTANCE_THRESHOLD = float(os.environ.get("KNOWN_DISTANCE_THRESHOLD", 0.8))

def _is_embedding_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".npy") or lower.endswith(".npz")


def _load_embedding_file(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "emb" in data.files:
            arr = data["emb"]
        elif data.files:
            arr = data[data.files[0]]
        else:
            raise RuntimeError(f"No arrays stored in {path}")
    else:
        arr = data
    return np.asarray(arr, dtype=float)


def extract_embedding(img_path: str) -> Tuple[bool, str | None, float | None]:
    """Return (known, label, distance) for the stored face image or embedding."""
    if _is_embedding_file(img_path):
        emb = _load_embedding_file(img_path)
        return is_known(emb, threshold=KNOWN_DISTANCE_THRESHOLD)

    img = Image.open(img_path).convert("RGB")
    tensors = detect_and_align(img, device="cpu")
    if not tensors:
        w, h = img.size
        mtcnn = get_mtcnn(device="cpu")
        tensors = mtcnn.extract(img, [(0, 0, w, h)], None) if mtcnn else []
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
    if not tensors:
        raise RuntimeError(f"No face detected in {img_path}")
    emb = embedding_from_face_tensor(tensors[0], device="cpu")
    return is_known(emb, threshold=KNOWN_DISTANCE_THRESHOLD)


def main() -> None:
    session = SessionLocal()
    faces = session.query(Face).all()
    session.close()
    if not faces:
        print("No faces in DB")
        return
    total = len(faces)
    failures = []
    for face in faces:
        try:
            known, label, dist = extract_embedding(face.image_path)
            status = "OK" if known else "UNKNOWN"
            print(f"{face.name:20s} -> {status:8s} label={label} dist={dist}")
            if not known:
                failures.append(face.name)
        except Exception as exc:
            print(f"{face.name:20s} -> ERROR {exc}")
            failures.append(face.name)
    print("-" * 40)
    if failures:
        print(f"{len(failures)} of {total} faces are not recognized: {failures}")
    else:
        print(f"All {total} faces recognized with threshold {KNOWN_DISTANCE_THRESHOLD}")


if __name__ == "__main__":
    main()
