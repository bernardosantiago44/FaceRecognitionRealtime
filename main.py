# main.py
import cv2
import face_recognition
import numpy as np
import queue
import threading
import time

from identity_store import IdentityStore
from in_memory_index import InMemoryIndex
from unknown_tracker import UnknownTracker
from registration_worker import RegistrationWorker, RegistrationJob, UpdateJob
from quality import blur_score

# --- NEW: UI imports ---
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from CameraSelectionDialog import CameraSelectionDialog
from threading import Thread
from queue import Queue, Empty

T_KNOWN = 0.5
BLUR_THRESHOLD = 100.0
BLUR_MARGIN_UPGRADE = 50.0  # how much sharper than threshold to trigger upgrade
DETECT_SCALE = 0.5
DETECT_EVERY_K = 5
TRACK_IOU_THRESHOLD = 0.3
MAX_TRACK_MISSED = 10
RECOG_COOLDOWN = 1.5  # seconds

recognition_q = Queue(maxsize=1)
results_q = Queue()

def find_best_match(embedding: np.ndarray, known_embeddings: np.ndarray):
    if known_embeddings.size == 0:
        return None, None
    diffs = known_embeddings - embedding.reshape(1, -1)
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def compute_iou(box_a, box_b):
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


def submit_recognition(payload):
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

            results_q.put((track_id, label, is_known, embedding))


Thread(target=recognition_worker, daemon=True).start()

def list_available_cameras(max_index: int = 5):
    """Return a list of camera indices that can be opened."""
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        if not ok:
            cap.release()
            continue
        cap.release()
        if ok:
            available.append(idx)
    return available


class FaceRecognitionApp:
    def __init__(self, root, camera_index: int = 0):
        self.root = root
        self.root.title("Face Recognition")

        # --- Core components (unchanged logic) ---
        self.store = IdentityStore(root_dir="identities")
        self.index = InMemoryIndex(self.store)

        self.job_queue: queue.Queue = queue.Queue(maxsize=512)
        self.stop_event = threading.Event()

        self.worker = RegistrationWorker(
            job_queue=self.job_queue,
            store=self.store,
            index=self.index,
            t_known=T_KNOWN,
            margin=0.05,
            blur_threshold=BLUR_THRESHOLD,
            max_images=3,
            stop_event=self.stop_event,
        )
        self.worker.start()

        self.tracker = UnknownTracker(
            iou_threshold=0.3,
            min_frames=5,
            max_inactive_frames=15,
            max_images=3,
            blur_threshold=BLUR_THRESHOLD,
        )

        self.tracks = {}
        self.next_track_id = 0
        self.frame_idx = 0

        # --- Camera setup ---
        self.camera_index = camera_index
        self.cap = None
        self._open_camera(self.camera_index)

        # --- UI layout ---
        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.select_cam_button = tk.Button(
            self.bottom_frame,
            text="Select Camera",
            command=self.change_camera
        )
        self.select_cam_button.pack(side=tk.BOTTOM, pady=5)

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start video loop
        self.update_frame()

    def _open_camera(self, index: int):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not open camera index {index}.")
            self.cap = None

    def _process_recognition_results(self, frame, frame_index):
        unknown_faces = []
        try:
            while True:
                track_id, label, is_known, embedding = results_q.get_nowait()
                track = self.tracks.get(track_id)
                if track is None:
                    continue

                pending = track.pop("pending", None)
                bbox = pending.get("bbox") if pending else track["bbox"]
                face_crop = pending.get("face_crop") if pending else None
                crop_blur = pending.get("blur", 0.0) if pending else 0.0

                if (face_crop is None or face_crop.size == 0) and bbox:
                    top, right, bottom, left = bbox
                    face_crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
                    crop_blur = blur_score(face_crop) if face_crop.size > 0 else 0.0
                else:
                    top, right, bottom, left = bbox

                track["label"] = label
                track["state"] = "known" if is_known else "unknown"

                meta = {}
                if is_known and label not in (None, "Unknown"):
                    meta = self.index.get_meta(label)
                track["display_name"] = meta.get("display_name") if meta else None

                if is_known and embedding is not None:
                    quality = meta.get("quality", {})
                    if quality.get("needs_refresh", False):
                        if crop_blur >= BLUR_THRESHOLD + BLUR_MARGIN_UPGRADE and face_crop is not None and face_crop.size > 0:
                            sample = {
                                "image": face_crop,
                                "blur": crop_blur,
                                "area": (right - left) * (bottom - top),
                            }
                            job = UpdateJob(identity_id=label, embedding=embedding, sample=sample)
                            try:
                                self.job_queue.put_nowait(job)
                            except queue.Full:
                                pass
                else:
                    if face_crop is not None and face_crop.size > 0 and embedding is not None:
                        unknown_faces.append(
                            {
                                "bbox": (left, top, right, bottom),
                                "embedding": embedding,
                                "face_image_bgr": face_crop,
                                "blur": crop_blur,
                            }
                        )
        except Empty:
            pass

        return unknown_faces

    def _update_tracks(self, detections):
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        matches = []

        while unmatched_tracks and unmatched_detections:
            best_pair = None
            best_iou = 0.0
            for track_id in unmatched_tracks:
                track_box = self.tracks[track_id]["bbox"]
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
            track = self.tracks[track_id]
            track["bbox"] = detections[det_idx]
            track["missed"] = 0

        for det_idx in unmatched_detections:
            self.tracks[self.next_track_id] = {
                "bbox": detections[det_idx],
                "label": None,
                "state": "loading",
                "display_name": None,
                "last_recog_time": 0.0,
                "missed": 0,
            }
            self.next_track_id += 1

        to_delete = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track["missed"] += 1
            if track["missed"] > MAX_TRACK_MISSED:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

    def change_camera(self):
        # Scan for available cameras
        available = list_available_cameras(max_index=5)
        if not available:
            messagebox.showerror("No Cameras", "No available cameras were found.")
            return

        # Show selection dialog (modal)
        dialog = CameraSelectionDialog(self.root, available, current_index=self.camera_index)
        # IMPORTANT: wait until the dialog is closed so .result is set
        self.root.wait_window(dialog)

        selected_idx = dialog.result
        if selected_idx is None:
            return  # user cancelled
        if selected_idx == self.camera_index:
            return  # no change

        # Switch to the selected camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_index = selected_idx
        self._open_camera(self.camera_index)


    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            # Try reopening current camera index in case it was temporarily unavailable
            self._open_camera(self.camera_index)
            self.root.after(30, self.update_frame)
            return

        success, frame = self.cap.read()
        if not success:
            self.root.after(30, self.update_frame)
            return

        frame_index = self.frame_idx
        unknown_faces = self._process_recognition_results(frame, frame_index)

        if frame_index % DETECT_EVERY_K == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations_small = face_recognition.face_locations(
                small_frame_rgb,
                number_of_times_to_upsample=1,
                model="hog"
            )
            detections = [
                (
                    int(top / DETECT_SCALE),
                    int(right / DETECT_SCALE),
                    int(bottom / DETECT_SCALE),
                    int(left / DETECT_SCALE),
                )
                for (top, right, bottom, left) in face_locations_small
            ]
            self._update_tracks(detections)
            recognition_tracks = []
            for track_id, track in self.tracks.items():
                last_recog = track.get("last_recog_time", 0.0)
                now = time.time()
                needs_recog = track.get("label") is None or (now - last_recog) > RECOG_COOLDOWN
                if not needs_recog:
                    continue
                top, right, bottom, left = track["bbox"]
                face_crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
                if face_crop.size == 0:
                    continue
                crop_blur = blur_score(face_crop)
                track["pending"] = {
                    "face_crop": face_crop,
                    "blur": crop_blur,
                    "bbox": track["bbox"],
                }
                track["state"] = "loading"
                track["last_recog_time"] = now
                recognition_tracks.append(
                    {
                        "track_id": track_id,
                        "face_crop": face_crop,
                    }
                )

            if recognition_tracks:
                ids, known_embs = self.index.get_snapshot()
                submit_recognition(
                    {
                        "tracks": recognition_tracks,
                        "ids": ids,
                        "known_embs": known_embs,
                    }
                )

        for track in self.tracks.values():
            top, right, bottom, left = track["bbox"]
            base_label = track.get("label")
            display_label = track.get("display_name") or base_label or "loading..."
            color = (0, 255, 0) if base_label not in (None, "Unknown") else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame,
                display_label,
                (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        self.frame_idx += 1

        # handle unknowns
        if unknown_faces:
            candidates = self.tracker.update(self.frame_idx, unknown_faces)
            for cand in candidates:
                job = RegistrationJob(candidate=cand)
                try:
                    self.job_queue.put_nowait(job)
                except queue.Full:
                    pass

        # --- Convert frame to Tkinter-compatible image and show it ---
        # OpenCV is BGR; convert to RGB
        frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb_display)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        # Keep a reference to avoid garbage collection
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule next frame
        self.root.after(10, self.update_frame)

    def on_close(self):
        # Clean up resources
        if self.cap is not None:
            self.cap.release()

        self.stop_event.set()
        try:
            self.job_queue.put_nowait(None)
        except queue.Full:
            pass

        self.worker.join(timeout=2)

        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root, camera_index=0)
    root.mainloop()


if __name__ == "__main__":
    main()
