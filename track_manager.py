# track_manager.py
# Track management utilities for the face recognition application.

from config import TRACK_IOU_THRESHOLD, MAX_TRACK_MISSED


def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two bounding boxes.
    
    Boxes are in format (top, right, bottom, left).
    """
    top_a, right_a, bottom_a, left_a = box_a
    top_b, right_b, bottom_b, left_b = box_b

    inter_left = max(left_a, left_b)
    inter_top = max(top_a, top_b)
    inter_right = min(right_a, right_b)
    inter_bottom = min(bottom_a, bottom_b)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area_a = (right_a - left_a) * (bottom_a - top_a)
    area_b = (right_b - left_b) * (bottom_b - top_b)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def update_tracks(tracks, detections, next_track_id):
    """Update tracks with new detections.
    
    Returns tuple of (updated_next_track_id).
    """
    unmatched_tracks = set(tracks.keys())
    unmatched_detections = set(range(len(detections)))
    matches = []

    while unmatched_tracks and unmatched_detections:
        best_pair = None
        best_iou = 0.0
        for track_id in unmatched_tracks:
            track_box = tracks[track_id]["bbox"]
            for det_idx in unmatched_detections:
                iou = compute_iou(track_box, detections[det_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (track_id, det_idx)

        if best_pair is None or best_iou < TRACK_IOU_THRESHOLD:
            break

        track_id, det_idx = best_pair
        matches.append((track_id, det_idx))
        unmatched_tracks.remove(track_id)
        unmatched_detections.remove(det_idx)

    for track_id, det_idx in matches:
        track = tracks[track_id]
        track["bbox"] = detections[det_idx]
        track["missed"] = 0

    for det_idx in unmatched_detections:
        tracks[next_track_id] = {
            "bbox": detections[det_idx],
            "label": None,
            "state": "loading",
            "display_name": None,
            "last_recog_time": 0.0,
            "missed": 0,
        }
        next_track_id += 1

    to_delete = []
    for track_id in unmatched_tracks:
        track = tracks[track_id]
        track["missed"] += 1
        if track["missed"] > MAX_TRACK_MISSED:
            to_delete.append(track_id)

    for track_id in to_delete:
        del tracks[track_id]

    return next_track_id
