import requests
BASE='http://127.0.0.1:5000'
url = BASE + '/vapid_public_key'
try:
    r = requests.get(url, timeout=10)
    print('status', r.status_code)
    print(r.json())
except Exception as e:
    print('request failed', e)
