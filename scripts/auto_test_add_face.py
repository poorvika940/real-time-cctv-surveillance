import requests, base64, time, io, sys
from pprint import pprint

BASE = 'http://127.0.0.1:5000'

def safe_get(path):
    try:
        r = requests.get(BASE + path, timeout=10)
        return r
    except Exception as e:
        print('GET', path, 'failed:', e)
        return None


def safe_post(path, data=None, files=None):
    try:
        r = requests.post(BASE + path, data=data, files=files, timeout=60)
        return r
    except Exception as e:
        print('POST', path, 'failed:', e)
        return None


print('Step 1: probe /debug_detect')
r = safe_get('/debug_detect')
if r is None:
    print('Server not reachable at', BASE)
    sys.exit(2)

try:
    j = r.json()
except Exception as e:
    print('Invalid JSON from /debug_detect:', e)
    print(r.text[:400])
    sys.exit(2)

print('debug_detect -> success:', j.get('success'))
faces = j.get('faces', []) if j.get('success') else []

if not faces:
    print('No faces from /debug_detect, trying /snapshot_faces')
    r2 = safe_post('/snapshot_faces', data={'camera': 0})
    if r2 is None:
        print('snapshot_faces request failed')
        sys.exit(2)
    try:
        j2 = r2.json()
    except Exception as e:
        print('snapshot_faces json fail', e)
        print(r2.text[:400])
        sys.exit(2)
    print('snapshot_faces ->', j2.get('success'))
    faces = j2.get('faces', []) if j2.get('success') else []

if not faces:
    print('No faces available to add. Make sure the camera shows a face.')
    sys.exit(3)

# pick the first face and its thumbnail
face = faces[0]
thumb_b64 = face.get('thumb') or face.get('image')
if not thumb_b64:
    # attempt to call debug_alert to get an image
    print('No thumbnail in face payload - calling /debug_alert to capture one')
    rdbg = safe_post('/debug_alert', data={'camera': 0, 'label': 'ManualCapture'})
    if rdbg:
        try:
            print('debug_alert ->', rdbg.json())
        except:
            print('debug_alert raw:', rdbg.text[:200])
    # re-call debug_detect
    r = safe_get('/debug_detect')
    try:
        j = r.json()
        faces = j.get('faces', [])
    except:
        faces = []
    if not faces:
        print('Still no faces after debug_alert. Aborting')
        sys.exit(4)
    face = faces[0]
    thumb_b64 = face.get('thumb') or face.get('image')

if not thumb_b64:
    print('No thumbnail available for the detected face. Aborting.')
    sys.exit(5)

# Prepare image bytes
try:
    img_bytes = base64.b64decode(thumb_b64)
except Exception as e:
    print('Failed to decode base64 image:', e)
    sys.exit(6)

name = f'AutoTest_{int(time.time())}'
prof = 'Student'
print(f'Uploading face as name={name}')
files = {'image': ('face.jpg', io.BytesIO(img_bytes), 'image/jpeg')}
resp = safe_post('/add_face', data={'name': name, 'profession': prof}, files=files)
if resp is None:
    print('/add_face failed')
    sys.exit(7)
try:
    jadd = resp.json()
except Exception:
    print('add_face response:', resp.text[:400])
    sys.exit(7)
print('/add_face ->', jadd)
if not jadd.get('success'):
    print('add_face unsuccessful, aborting')
    sys.exit(8)

print('Calling /retrain_sync (blocking) ...')
r = safe_post('/retrain_sync')
if r is None:
    print('/retrain_sync failed')
    sys.exit(9)
try:
    jr = r.json()
except Exception:
    print('/retrain_sync raw:', r.text[:400])
    jr = None
print('/retrain_sync ->', jr)

print('Waiting 1s then probing /debug_detect to check recognition...')
time.sleep(1.2)
r = safe_get('/debug_detect')
if r:
    try:
        jd = r.json()
        print('/debug_detect after retrain -> success=', jd.get('success'))
        pprint(jd.get('faces'))
        # check for known match
        for f in (jd.get('faces') or []):
            print('face known=', f.get('known'), 'label=', f.get('label'), 'distance=', f.get('distance'))
    except Exception as e:
        print('debug_detect JSON parse failed:', e)
        print(r.text[:400])

print('Done')
