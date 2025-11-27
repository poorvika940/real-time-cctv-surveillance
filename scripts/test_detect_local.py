#!/usr/bin/env python3
"""Quick local tester: run MTCNN detection and FaceNet embedding on a given image file.
Usage:
  .\.venv312\Scripts\Activate.ps1
  python scripts\test_detect_local.py <path-to-image>

Prints detected boxes, probabilities, and classification results.
"""
import sys
import os
import base64
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import get_mtcnn, detect_and_align, embedding_from_face_tensor, is_known


def run(path):
    if not os.path.exists(path):
        print('File not found:', path); return
    img = Image.open(path).convert('RGB')
    print('Running MTCNN detect + align...')
    face_tensors = detect_and_align(img, device='cpu')
    if not face_tensors:
        print('No faces detected')
        return
    print(f'Extracted {len(face_tensors)} aligned face tensor(s)')
    for i, ft in enumerate(face_tensors):
        # Ensure ft is a 3D tensor (C,H,W). Some mtcnn versions return tensors with extra batch dims.
        try:
            import torch
            if isinstance(ft, torch.Tensor):
                while ft.dim() > 3:
                    ft = ft.squeeze(0)
        except Exception:
            pass
        emb = embedding_from_face_tensor(ft, device='cpu')
        known, label, dist = is_known(emb)
        print(f'Face {i}: known={known} label={label} dist={dist}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_detect_local.py <image>')
    else:
        run(sys.argv[1])
