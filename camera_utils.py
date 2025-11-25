# camera_utils.py
# Camera utilities

import cv2


def list_available_cameras(max_index: int = 5):
    """Return a list of camera indices that can be opened."""
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if ok:
            available.append(idx)
    return available
