#!/usr/bin/env python3
"""Generate VAPID keypair (P-256) and save to models/vapid.json as base64url keys.

Requires: cryptography

Usage:
    .\.venv312\Scripts\Activate.ps1
    pip install cryptography
    python scripts\generate_vapid.py

This will write models/vapid.json with fields `publicKey` and `privateKey`.
"""
import os
import json
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VAPID_PATH = os.path.join(MODEL_DIR, 'vapid.json')

os.makedirs(MODEL_DIR, exist_ok=True)

# generate private key (P-256)
priv = ec.generate_private_key(ec.SECP256R1())
priv_numbers = priv.private_numbers()
priv_value = priv_numbers.private_value
priv_bytes = priv_value.to_bytes(32, 'big')
# public key uncompressed format (0x04 | X | Y)
pub = priv.public_key()
pub_numbers = pub.public_numbers()
x = pub_numbers.x.to_bytes(32, 'big')
y = pub_numbers.y.to_bytes(32, 'big')
uncompressed = b'\x04' + x + y

public_key_b64 = base64.urlsafe_b64encode(uncompressed).rstrip(b'=') .decode('utf-8')
private_key_b64 = base64.urlsafe_b64encode(priv_bytes).rstrip(b'=') .decode('utf-8')

vapid = {'publicKey': public_key_b64, 'privateKey': private_key_b64}
with open(VAPID_PATH, 'w', encoding='utf-8') as f:
    json.dump(vapid, f, indent=2)

print('Saved VAPID keys to', VAPID_PATH)
print('Public key (base64url):', public_key_b64)
