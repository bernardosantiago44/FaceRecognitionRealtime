# matching.py
# Face matching utilities

import numpy as np


def find_best_match(embedding: np.ndarray, known_embeddings: np.ndarray):
    if known_embeddings.size == 0:
        return None, None
    diffs = known_embeddings - embedding.reshape(1, -1)
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])
