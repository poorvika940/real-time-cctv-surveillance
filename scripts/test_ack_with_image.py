import requests, base64, io
from PIL import Image

BASE='http://127.0.0.1:5000'
url = BASE + '/ack_alert'

# create a simple gray face-like image 160x160
img = Image.new('RGB', (160,160), color=(120,120,120))
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_b = buf.getvalue()
img_b64 = base64.b64encode(img_b).decode('utf-8')

payload = {'camera': '0', 'name': 'E2ETest', 'profession': 'Student', 'image_b64': img_b64}
print('Posting to', url)
try:
    r = requests.post(url, json=payload, timeout=30)
    print('status', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text[:500])
except Exception as e:
    print('Request failed', e)
