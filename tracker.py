import time
import numpy as np

def iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=5, alert_cooldown=5.0):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.alert_cooldown = alert_cooldown
        self.next_id = 1
        self.tracks = {}  # id -> dict(bbox, missed, last_seen, created, last_alert)

    def _create_track(self, bbox):
        tid = self.next_id
        self.next_id += 1
        now = time.time()
        self.tracks[tid] = {'bbox': bbox, 'missed': 0, 'last_seen': now, 'created': now, 'last_alert': 0}
        return tid

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2]
        Returns list of track dicts corresponding to each detection in order: {'id':id,'bbox':bbox,'track':track_state}
        """
        results = [None] * len(detections)
        if len(detections) == 0:
            # increment missed for all tracks and cleanup
            remove = []
            for tid, t in self.tracks.items():
                t['missed'] += 1
                if t['missed'] > self.max_missed:
                    remove.append(tid)
            for r in remove:
                del self.tracks[r]
            return []

        track_ids = list(self.tracks.keys())
        if len(track_ids) == 0:
            # create tracks for all detections
            for i, d in enumerate(detections):
                tid = self._create_track(d)
                results[i] = {'id': tid, 'bbox': d, 'track': self.tracks[tid]}
            return results

        # compute IoU matrix
        iou_mat = np.zeros((len(track_ids), len(detections)), dtype=float)
        for ti, tid in enumerate(track_ids):
            for di, d in enumerate(detections):
                iou_mat[ti, di] = iou(self.tracks[tid]['bbox'], d)

        assigned_tracks = set()
        assigned_dets = set()

        # greedy assignment: pick highest IoU pairs
        while True:
            idx = np.argmax(iou_mat)
            ti, di = divmod(int(idx), iou_mat.shape[1])
            max_iou = iou_mat[ti, di]
            if max_iou < self.iou_threshold:
                break
            tid = track_ids[ti]
            if tid in assigned_tracks or di in assigned_dets:
                iou_mat[ti, di] = -1
                continue
            # assign
            self.tracks[tid]['bbox'] = detections[di]
            self.tracks[tid]['missed'] = 0
            self.tracks[tid]['last_seen'] = time.time()
            results[di] = {'id': tid, 'bbox': detections[di], 'track': self.tracks[tid]}
            assigned_tracks.add(tid)
            assigned_dets.add(di)
            # invalidate row and col
            iou_mat[ti, :] = -1
            iou_mat[:, di] = -1

        # create tracks for unassigned detections
        for di, d in enumerate(detections):
            if di in assigned_dets:
                continue
            tid = self._create_track(d)
            results[di] = {'id': tid, 'bbox': d, 'track': self.tracks[tid]}

        # increment missed for unassigned tracks
        for tid, t in list(self.tracks.items()):
            if tid not in assigned_tracks:
                t['missed'] += 1
                if t['missed'] > self.max_missed:
                    del self.tracks[tid]

        return results

    def should_alert(self, track):
        now = time.time()
        last = track.get('last_alert', 0)
        return (now - last) >= self.alert_cooldown

    def mark_alerted(self, track):
        track['last_alert'] = time.time()
