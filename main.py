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

# --- NEW: UI imports ---
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from CameraSelectionDialog import CameraSelectionDialog

T_KNOWN = 0.5
BLUR_THRESHOLD = 100.0
BLUR_MARGIN_UPGRADE = 50.0  # how much sharper than threshold to trigger upgrade
SCALE = 0.5


def find_best_match(embedding: np.ndarray, known_embeddings: np.ndarray):
    if known_embeddings.size == 0:
        return None, None
    diffs = known_embeddings - embedding.reshape(1, -1)
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])

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

        self.frame_idx += 1

        # --- Main recognition logic (adapted from your original loop) ---
        img_small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            img_small_rgb,
            number_of_times_to_upsample=1,
            model="hog"
        )
        face_encodings = face_recognition.face_encodings(img_small_rgb, face_locations)

        ids, known_embs = self.index.get_snapshot()
        unknown_faces = []

        for (top, right, bottom, left), face_embedding in zip(face_locations, face_encodings):
            emb = np.array(face_embedding, dtype="float32")

            # upscale box
            scale = 1 / SCALE
            top = int(top * scale)
            right = int(right * scale)
            bottom = int(bottom * scale)
            left = int(left * scale)

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
                    meta = self.index.get_meta(identity_id)
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
                                self.job_queue.put_nowait(job)
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
            candidates = self.tracker.update(self.frame_idx, unknown_faces)
            for cand in candidates:
                job = RegistrationJob(candidate=cand)
                try:
                    self.job_queue.put_nowait(job)
                except queue.Full:
                    pass

        # --- Convert frame to Tkinter-compatible image and show it ---
        # OpenCV is BGR; convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
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
    app = FaceRecognitionApp(root, camera_index=1)
    root.mainloop()


if __name__ == "__main__":
    main()
