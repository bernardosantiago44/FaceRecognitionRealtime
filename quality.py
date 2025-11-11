# quality.py

import cv2
import numpy as np

def blur_score(bgr_img: np.ndarray) -> float:
    if bgr_img is None or bgr_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
