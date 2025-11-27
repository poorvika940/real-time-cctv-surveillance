import requests, sys, os
BASE='http://127.0.0.1:5000'
img_path = sys.argv[1] if len(sys.argv)>1 else 'dataset/Poorvika_1762692866.jpg'
name = sys.argv[2] if len(sys.argv)>2 else 'AutoPostTest'
prof = sys.argv[3] if len(sys.argv)>3 else 'Student'
if not os.path.exists(img_path):
    print('Image not found', img_path); sys.exit(2)
with open(img_path,'rb') as f:
    files={'image': (os.path.basename(img_path), f, 'image/jpeg')}
    data={'name': name, 'profession': prof}
    r = requests.post(BASE + '/add_face', data=data, files=files, timeout=60)
    print('status', r.status_code)
    try:
        print(r.json())
    except Exception:
        print('resp:', r.text[:400])
