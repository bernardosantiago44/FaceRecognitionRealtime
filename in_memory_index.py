# in_memory_index.py

import threading
from typing import List, Tuple, Dict

import numpy as np

from identity_store import IdentityStore


class InMemoryIndex:
    def __init__(self, store: IdentityStore):
        self._store = store
        self._lock = threading.RLock()
        ids, embs, metas = self._store.load_all_embeddings_and_meta()
        self._ids: List[str] = ids
        self._embeddings: np.ndarray = embs
        self._meta: Dict[str, dict] = metas

    def get_snapshot(self) -> Tuple[List[str], np.ndarray]:
        with self._lock:
            return list(self._ids), self._embeddings.copy()

    def get_meta(self, identity_id: str) -> dict:
        with self._lock:
            return dict(self._meta.get(identity_id, {}))

    def add_identity(self, identity_id: str, embedding: np.ndarray, meta: dict):
        if embedding.ndim != 1:
            raise ValueError("embedding must be 1D")
        with self._lock:
            self._ids.append(identity_id)
            if self._embeddings.size == 0:
                self._embeddings = embedding.reshape(1, -1).astype("float32")
            else:
                self._embeddings = np.vstack(
                    [self._embeddings, embedding.reshape(1, -1)]
                ).astype("float32")
            self._meta[identity_id] = meta

    def update_embedding(self, identity_id: str, new_embedding: np.ndarray):
        if new_embedding.ndim != 1:
            raise ValueError("new_embedding must be 1D")
        with self._lock:
            if identity_id not in self._ids:
                return
            idx = self._ids.index(identity_id)
            self._embeddings[idx, :] = new_embedding.astype("float32")

    def update_meta(self, identity_id: str, meta: dict):
        with self._lock:
            self._meta[identity_id] = dict(meta)
