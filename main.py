# main.py

import cv2
import face_recognition
import numpy as np
import queue
import threading

from identity_store import IdentityStore
from in_memory_index import InMemoryIndex
from unknown_tracker import UnknownTracker
from registration_worker import RegistrationWorker, RegistrationJob, UpdateJob
from quality import blur_score


T_KNOWN = 0.5
BLUR_THRESHOLD = 100.0
BLUR_MARGIN_UPGRADE = 50.0  # how much sharper than threshold to trigger upgrade


def find_best_match(embedding: np.ndarray, known_embeddings: np.ndarray):
    if known_embeddings.size == 0:
        return None, None
    diffs = known_embeddings - embedding.reshape(1, -1)
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def main():
    store = IdentityStore(root_dir="identities")
    index = InMemoryIndex(store)

    job_queue: queue.Queue = queue.Queue(maxsize=512)
    stop_event = threading.Event()

    worker = RegistrationWorker(
        job_queue=job_queue,
        store=store,
        index=index,
        t_known=T_KNOWN,
        margin=0.05,
        blur_threshold=BLUR_THRESHOLD,
        max_images=3,
        stop_event=stop_event,
    )
    worker.start()

    tracker = UnknownTracker(
        iou_threshold=0.3,
        min_frames=5,
        max_inactive_frames=15,
        max_images=3,
        blur_threshold=BLUR_THRESHOLD,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_idx = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1

            img_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(img_small_rgb)
            face_encodings = face_recognition.face_encodings(img_small_rgb, face_locations)

            ids, known_embs = index.get_snapshot()
            unknown_faces = []

            for (top, right, bottom, left), face_embedding in zip(face_locations, face_encodings):
                emb = np.array(face_embedding, dtype="float32")

                # upscale box
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # crop
                face_crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
                crop_blur = blur_score(face_crop) if face_crop.size > 0 else 0.0

                label = "Unknown"
                color = (0, 0, 255)

                if known_embs.size > 0:
                    idx_match, dist = find_best_match(emb, known_embs)
                    if idx_match is not None and dist is not None and dist <= T_KNOWN:
                        identity_id = ids[idx_match]
                        label = identity_id
                        color = (0, 255, 0)

                        # upgrade path if this identity was low-quality
                        meta = index.get_meta(identity_id)
                        quality = meta.get("quality", {})
                        if quality.get("needs_refresh", False):
                            if crop_blur >= BLUR_THRESHOLD + BLUR_MARGIN_UPGRADE and face_crop.size > 0:
                                sample = {
                                    "image": face_crop,
                                    "blur": crop_blur,
                                    "area": (right - left) * (bottom - top),
                                }
                                job = UpdateJob(identity_id=identity_id, embedding=emb, sample=sample)
                                try:
                                    job_queue.put_nowait(job)
                                except queue.Full:
                                    pass
                    else:
                        # unknown
                        if face_crop.size > 0:
                            unknown_faces.append(
                                {
                                    "bbox": (left, top, right, bottom),
                                    "embedding": emb,
                                    "face_image_bgr": face_crop,
                                    "blur": crop_blur,
                                }
                            )
                else:
                    # no known embeddings yet
                    if face_crop.size > 0:
                        unknown_faces.append(
                            {
                                "bbox": (left, top, right, bottom),
                                "embedding": emb,
                                "face_image_bgr": face_crop,
                                "blur": crop_blur,
                            }
                        )

                # draw
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    label,
                    (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # handle unknowns
            if unknown_faces:
                candidates = tracker.update(frame_idx, unknown_faces)
                for cand in candidates:
                    job = RegistrationJob(candidate=cand)
                    try:
                        job_queue.put_nowait(job)
                    except queue.Full:
                        pass

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_event.set()
        try:
            job_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
