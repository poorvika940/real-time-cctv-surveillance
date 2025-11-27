#!/usr/bin/env python3
"""Add an existing image file to the DB as a labeled face entry.
Usage:
  .\.venv312\Scripts\Activate.ps1
  python scripts\add_face_entry.py "dataset/Vinuth/IMG-20251010-WA0016[1].jpg" "Vinuth" "Student"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import SessionLocal, init_db, Face

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: add_face_entry.py <image_path> <name> [profession]')
        sys.exit(1)
    img = sys.argv[1]
    name = sys.argv[2]
    prof = sys.argv[3] if len(sys.argv) > 3 else ''
    if not os.path.exists(img):
        print('Image not found:', img); sys.exit(1)
    img_abs = os.path.abspath(img)
    init_db()
    session = SessionLocal()
    f = Face(name=name, profession=prof, image_path=img_abs)
    session.add(f)
    session.commit()
    print('Added face id', f.id, 'name', name, 'image', img_abs)
    session.close()
