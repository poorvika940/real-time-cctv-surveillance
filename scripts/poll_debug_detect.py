import requests, time, json
BASE='http://127.0.0.1:5000'
url = BASE + '/debug_detect?camera=0'
for i in range(6):
    print(f'Iteration {i+1}/6')
    try:
        r = requests.get(url, timeout=10)
        print('status', r.status_code)
        j = r.json()
        if not j.get('success'):
            print('error:', j.get('error'))
        else:
            faces = j.get('faces', [])
            print(f'faces: {len(faces)}')
            for fi, f in enumerate(faces):
                print(f"  #{fi+1}: known={f.get('known')} label={f.get('label')} dist={f.get('distance')}")
    except Exception as e:
        print('request failed', e)
    if i < 5:
        time.sleep(2)
print('done')
