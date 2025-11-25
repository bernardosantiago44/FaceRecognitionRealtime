# recognition.py
# Face recognition worker logic for the face recognition application.

import cv2
import face_recognition
import logging
import numpy as np
from queue import Queue, Empty
from threading import Thread

from config import T_KNOWN

logger = logging.getLogger(__name__)

recognition_q = Queue(maxsize=1)
results_q = Queue()


def find_best_match(embedding: np.ndarray, known_embeddings: np.ndarray):
    """Find the best matching embedding from known embeddings."""
    if known_embeddings.size == 0:
        return None, None
    diffs = known_embeddings - embedding.reshape(1, -1)
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def submit_recognition(payload):
    """Submit a recognition payload to the recognition queue."""
    if recognition_q.full():
        try:
            recognition_q.get_nowait()
        except Empty:
            pass
    try:
        recognition_q.put_nowait(payload)
    except Empty:
        pass


def recognition_worker():
    """Background worker that processes face recognition tasks."""
    while True:
        payload = recognition_q.get()
        tracks_payload = payload.get("tracks", [])
        ids = payload.get("ids", [])
        known_embs = payload.get("known_embs", np.array([]))

        for track_payload in tracks_payload:
            track_id = track_payload["track_id"]
            face_crop = track_payload["face_crop"]
            embedding = None
            label = "Unknown"
            is_known = False

            if face_crop.size > 0:
                crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(crop_rgb)
                if encodings:
                    embedding = np.array(encodings[0], dtype="float32")
                    if known_embs.size > 0:
                        idx_match, dist = find_best_match(embedding, known_embs)
                        if idx_match is not None and dist is not None and dist <= T_KNOWN:
                            label = ids[idx_match]
                            is_known = True
                            logger.debug(f"Track {track_id}: matched to {label} (dist={dist:.3f})")
                        else:
                            logger.debug(f"Track {track_id}: unknown face (min_dist={dist:.3f if dist else 'N/A'})")
                    else:
                        logger.debug(f"Track {track_id}: unknown face (no known embeddings)")

            results_q.put((track_id, label, is_known, embedding))


def start_recognition_worker():
    """Start the background recognition worker thread."""
    Thread(target=recognition_worker, daemon=True).start()
