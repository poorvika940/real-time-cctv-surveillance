import requests, sys, json
BASE='http://127.0.0.1:5000'
cam = sys.argv[1] if len(sys.argv)>1 else '0'
track = sys.argv[2] if len(sys.argv)>2 else '42'
name = sys.argv[3] if len(sys.argv)>3 else 'ManualAck'
url = BASE + '/ack_alert'
print('Posting ack', cam, track, name)
r = requests.post(url, json={'camera': cam, 'track_id': track, 'name': name}, timeout=10)
print('status', r.status_code)
try:
    print(r.json())
except Exception:
    print(r.text)
