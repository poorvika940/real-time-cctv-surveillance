#!/usr/bin/env python3
"""Batch add face images to the SQLite DB one-by-one with an optional delay.

Supports two input modes:
- A CSV file with columns: image_path,name,profession
- A root folder: will walk subfolders and optionally infer the person's name from the folder name

Options:
  --csv CSV_FILE
  --root ROOT_DIR
  --pattern GLOB_PATTERN (default *.jpg)
  --delay SECONDS (float, default 1.0)
  --infer-name-from-folder (use the immediate parent folder name as the person's name)
  --profession DEFAULT_PROF (use a default profession for all entries when inferring)
  --retrain-after-each (bool) - if set, calls the /retrain endpoint after each add
  --retrain-at-end (bool) - if set, calls /retrain once after processing all images
  --server URL - base server URL (default http://localhost:5000) used when calling /retrain

Example:
  .\\.venv312\\Scripts\\Activate.ps1
  python scripts\\batch_add_faces.py --root dataset --infer-name-from-folder --delay 1.5 --retrain-at-end

"""
import argparse
import os
import time
import csv
import glob
import requests

# Add project root to path to import DB helpers
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import SessionLocal, init_db, Face


def add_face_to_db(image_path, name, profession=''):
    session = SessionLocal()
    f = Face(name=name, profession=profession, image_path=os.path.abspath(image_path))
    session.add(f)
    session.commit()
    fid = f.id
    session.close()
    return fid


def call_retrain(server_url):
    try:
        r = requests.post(server_url.rstrip('/') + '/retrain', timeout=10)
        try:
            return r.json()
        except Exception:
            return {'status_code': r.status_code}
    except Exception as e:
        return {'error': str(e)}


def process_csv(path, delay, retrain_each, retrain_at_end, server_url, default_prof):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            img = row[0].strip()
            name = row[1].strip()
            prof = row[2].strip() if len(row) > 2 else default_prof
            if not os.path.exists(img):
                print('SKIP missing image:', img)
                continue
            fid = add_face_to_db(img, name, prof)
            print('Added', img, '-> id', fid, 'name', name)
            if retrain_each:
                print('Triggering retrain...')
                print(call_retrain(server_url))
            time.sleep(delay)
    if retrain_at_end:
        print('Triggering retrain at end...')
        print(call_retrain(server_url))


def process_root(root, pattern, delay, infer_name, default_prof, retrain_each, retrain_at_end, server_url):
    # Walk folders; if infer_name True, use immediate parent folder name as label
    paths = sorted(glob.glob(os.path.join(root, '**', pattern), recursive=True))
    if not paths:
        print('No images found with pattern', pattern, 'under', root)
        return
    print('Found', len(paths), 'images')
    for p in paths:
        name = None
        prof = default_prof
        if infer_name:
            parent = os.path.basename(os.path.dirname(p))
            name = parent
        else:
            # fallback to file stem
            name = os.path.splitext(os.path.basename(p))[0]
        fid = add_face_to_db(p, name, prof)
        print('Added', p, '-> id', fid, 'name', name)
        if retrain_each:
            print('Triggering retrain...')
            print(call_retrain(server_url))
        time.sleep(delay)
    if retrain_at_end:
        print('Triggering retrain at end...')
        print(call_retrain(server_url))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='CSV file with image_path,name,profession')
    parser.add_argument('--root', help='Root folder to walk for images')
    parser.add_argument('--pattern', default='*.jpg', help='Glob pattern for images (default *.jpg)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay in seconds between adding entries')
    parser.add_argument('--infer-name-from-folder', action='store_true', help='Use parent folder name as person name')
    parser.add_argument('--profession', default='', help='Default profession when inferring names')
    parser.add_argument('--retrain-after-each', dest='retrain_each', action='store_true', help='Call /retrain after each add')
    parser.add_argument('--retrain-at-end', dest='retrain_at_end', action='store_true', help='Call /retrain after processing all images')
    parser.add_argument('--server', default='http://localhost:5000', help='Base URL for server endpoints (used for retrain)')
    args = parser.parse_args()

    if not args.csv and not args.root:
        print('Either --csv or --root must be provided')
        sys.exit(1)

    init_db()

    if args.csv:
        process_csv(args.csv, args.delay, args.retrain_each, args.retrain_at_end, args.server, args.profession)
    else:
        process_root(args.root, args.pattern, args.delay, args.infer_name_from_folder, args.profession, args.retrain_each, args.retrain_at_end, args.server)

    print('Done')
