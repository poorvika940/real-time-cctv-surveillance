"""
Evaluate model predictions recorded in models/predictions.csv
Prints overall accuracy, per-class accuracy and a confusion matrix, and lists misclassified examples.
Run with project's venv python:
& ".\\.venv312\\Scripts\\python.exe" evaluate_predictions.py
"""
import csv
import os
from collections import defaultdict

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, 'models', 'predictions.csv')

if not os.path.exists(CSV_PATH):
    print('No predictions CSV found at', CSV_PATH)
    raise SystemExit(2)

rows = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# Filter rows where true_label is present
total = len(rows)
detected = sum(1 for r in rows if r['predicted_label'])
correct = sum(1 for r in rows if r['predicted_label'] and r['predicted_label'] == r['true_label'])

# Per-class stats
classes = set(r['true_label'] for r in rows if r['true_label'])
per_class = {c: {'total':0, 'correct':0} for c in classes}
for r in rows:
    tl = r['true_label']
    pl = r['predicted_label']
    if tl in per_class:
        per_class[tl]['total'] += 1
        if pl and pl == tl:
            per_class[tl]['correct'] += 1

# Confusion matrix using classes sorted
labels = sorted(list(classes))
label_to_idx = {l:i for i,l in enumerate(labels)}
import numpy as np
cm = np.zeros((len(labels), len(labels)), dtype=int)
for r in rows:
    tl = r['true_label']
    pl = r['predicted_label']
    if tl not in label_to_idx:
        continue
    i = label_to_idx[tl]
    if pl and pl in label_to_idx:
        j = label_to_idx[pl]
        cm[i,j] += 1
    else:
        # treat as predicted as blank / unknown -> count in a special column beyond existing labels
        pass

print('Total images:', total)
print('Detected (predicted non-empty):', detected)
print('Correct predictions:', correct)
print('Overall accuracy (on detected):', (correct/detected) if detected else 'N/A')
print('\nPer-class accuracy:')
for c in labels:
    t = per_class[c]['total']
    corr = per_class[c]['correct']
    acc = (corr / t) if t else 0.0
    print(f"  {c}: {corr}/{t} = {acc:.2f}")

print('\nConfusion matrix (rows=true, cols=predicted)')
print('Labels order:', labels)
print(cm)

# Show some misclassified examples
mis = [r for r in rows if r['predicted_label'] and r['predicted_label'] != r['true_label']]
if mis:
    print('\nMisclassified examples:')
    for r in mis[:20]:
        print(f"  {r['image_path']} -> true={r['true_label']} predicted={r['predicted_label']} distance={r.get('distance')}")
else:
    print('\nNo misclassified examples where a prediction was made (others may have no face detected).')

# show images with no face detected
no_face = [r for r in rows if not r['predicted_label']]
if no_face:
    print(f"\nImages with no face detected: {len(no_face)} (showing up to 20)")
    for r in no_face[:20]:
        print('  ', r['image_path'])
