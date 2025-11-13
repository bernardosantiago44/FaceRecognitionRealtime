# registration_worker.py

import threading
import queue
from typing import Optional

import numpy as np

from identity_store import IdentityStore
from in_memory_index import InMemoryIndex
from unknown_tracker import RegistrationCandidate
from ResourcePath import resource_path


class RegistrationJob:
    def __init__(self, candidate: RegistrationCandidate):
        self.candidate = candidate


class UpdateJob:
    def __init__(self, identity_id: str, embedding: np.ndarray, sample: dict):
        self.identity_id = identity_id
        self.embedding = embedding
        self.sample = sample  # { "image":..., "blur":..., "area":... }


class RegistrationWorker(threading.Thread):
    def __init__(
        self,
        job_queue: queue.Queue,
        store: IdentityStore,
        index: InMemoryIndex,
        t_known: float,
        margin: float = 0.05,
        blur_threshold: float = 100.0,
        max_images: int = 3,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(daemon=True)
        self._queue = job_queue
        self._store = store
        self._index = index
        self._t_known = t_known
        self._margin = margin
        self._blur_threshold = blur_threshold
        self._max_images = max_images
        self._stop_event = stop_event or threading.Event()
        self._running = True

    def run(self):
        while self._running and not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if job is None:
                self._queue.task_done()
                break

            try:
                if isinstance(job, RegistrationJob):
                    self._handle_registration(job)
                elif isinstance(job, UpdateJob):
                    self._handle_update(job)
            finally:
                self._queue.task_done()

    def _handle_registration(self, job: RegistrationJob):
        cand = job.candidate
        emb = cand.embedding

        ids, known_embs = self._index.get_snapshot()
        if known_embs.size > 0:
            diffs = known_embs - emb.reshape(1, -1)
            dists = np.linalg.norm(diffs, axis=1)
            idx = int(np.argmin(dists))
            d_min = float(dists[idx])
            if d_min <= self._t_known + self._margin:
                return

        extra_meta = dict(cand.meta)
        identity_id = self._store.create_identity_from_samples(
            embedding=emb,
            samples=cand.samples,
            source=extra_meta.pop("source", "online_capture"),
            label=None,
            extra_meta=extra_meta,
            max_images=self._max_images,
        )

        # Load back meta to keep index consistent
        meta = self._store.load_meta(identity_id) or {}
        self._index.add_identity(identity_id, emb, meta)

    def _handle_update(self, job: UpdateJob):
        # Upgrade blurry identity when sharper sample arrives
        meta = self._store.update_identity_with_sample(
            identity_id=job.identity_id,
            new_embedding=job.embedding,
            sample=job.sample,
            max_images=self._max_images,
            blur_threshold=self._blur_threshold,
        )
        if not meta:
            return

        # Recompute representative embedding (mean) for index
        ident_dir = resource_path(f"{self._store.root_dir}/{job.identity_id}")
        emb_path = resource_path(f"{ident_dir}/embeddings.npz")
        try:
            data = np.load(emb_path)
            embs = data.get("embeddings")
            if embs is not None and embs.size > 0:
                rep = embs.mean(axis=0).astype("float32")
                self._index.update_embedding(job.identity_id, rep)
                self._index.update_meta(job.identity_id, meta)
        except Exception:
            return

    def stop(self):
        self._running = False
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
