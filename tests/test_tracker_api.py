import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app


def test_tracker_config_get_set(tmp_path):
    client = app.test_client()
    # GET default
    r = client.get('/tracker_config?camera=0')
    assert r.status_code == 200
    j = r.get_json()
    assert 'success' in j and j['success']

    # POST set values
    r = client.post('/tracker_config', data={'camera': '0', 'iou_threshold': '0.4', 'max_missed': '3', 'alert_cooldown': '2.5', 'show_ids': '1'})
    assert r.status_code == 200
    j = r.get_json()
    assert j['success'] and j['settings']['iou_threshold'] == 0.4