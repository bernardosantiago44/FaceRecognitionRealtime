# main.py
import cv2
import face_recognition
import json
import os
import queue
import threading
import time

from identity_store import IdentityStore
from in_memory_index import InMemoryIndex
from unknown_tracker import UnknownTracker
from registration_worker import RegistrationWorker, RegistrationJob, UpdateJob
from quality import blur_score
from ResourcePath import resource_path

# --- UI imports ---
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
from CameraSelectionDialog import CameraSelectionDialog
from queue import Empty

# --- Modular imports ---
from config import (
    T_KNOWN,
    BLUR_THRESHOLD,
    BLUR_MARGIN_UPGRADE,
    DETECT_SCALE,
    DETECT_EVERY_K,
    RECOG_COOLDOWN,
)
from camera_utils import list_available_cameras
from recognition import results_q, submit_recognition, start_recognition_worker
from track_manager import update_tracks

# Start the recognition worker thread
start_recognition_worker()


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

        # --- Label Mode state (session only) ---
        self.label_mode = False
        self.hovered_track_id = None

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
        self.select_cam_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Label Mode toggle button
        self.label_mode_button = tk.Button(
            self.bottom_frame,
            text="Label Mode: Off",
            command=self._toggle_label_mode,
            relief=tk.RAISED,
        )
        self.label_mode_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind click and motion events for Label Mode
        self.video_label.bind("<Button-1>", self._on_video_click)
        self.video_label.bind("<Motion>", self._on_video_motion)

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
        self.next_track_id = update_tracks(self.tracks, detections, self.next_track_id)

    def _toggle_label_mode(self):
        """Toggle Label Mode on/off."""
        self.label_mode = not self.label_mode
        if self.label_mode:
            self.label_mode_button.config(text="Label Mode: On", relief=tk.SUNKEN)
        else:
            self.label_mode_button.config(text="Label Mode: Off", relief=tk.RAISED)
            self.hovered_track_id = None

    def _find_track_at_position(self, x, y):
        """Find the track ID at the given (x, y) position, or None if no face is there."""
        for track_id, track in self.tracks.items():
            top, right, bottom, left = track["bbox"]
            if left <= x <= right and top <= y <= bottom:
                return track_id
        return None

    def _on_video_click(self, event):
        """Handle click on the video label."""
        if not self.label_mode:
            return

        track_id = self._find_track_at_position(event.x, event.y)
        if track_id is None:
            return

        track = self.tracks.get(track_id)
        if track is None:
            return

        # Get the current label/name for the dialog
        current_label = track.get("label")
        current_name = track.get("display_name") or ""

        self._show_naming_dialog(track_id, current_label, current_name)

    def _on_video_motion(self, event):
        """Handle mouse motion over the video label for hover highlighting."""
        if not self.label_mode:
            self.hovered_track_id = None
            return

        self.hovered_track_id = self._find_track_at_position(event.x, event.y)

    def _show_naming_dialog(self, track_id, identity_id, current_name):
        """Show a dialog to name/rename a face."""
        track = self.tracks.get(track_id)
        if track is None:
            return

        prompt = f"Enter name for {identity_id or 'this person'}:"
        new_name = simpledialog.askstring(
            "Name Person",
            prompt,
            initialvalue=current_name,
            parent=self.root
        )

        if new_name is None:
            return  # User cancelled

        # Update the display name in the track
        track["display_name"] = new_name if new_name.strip() else None

        # Persist the name if we have a valid identity_id
        if identity_id and identity_id not in (None, "Unknown"):
            self._update_person_name(identity_id, new_name)

    def _update_person_name(self, identity_id, new_name):
        """Update the display name for a person in the store."""
        ident_dir = resource_path(os.path.join(self.store.root_dir, identity_id))
        meta_path = resource_path(os.path.join(ident_dir, "meta.json"))

        if not os.path.exists(meta_path):
            return

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            meta["display_name"] = new_name.strip() if new_name and new_name.strip() else None

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Update the in-memory index
            self.index.update_meta(identity_id, meta)
        except (IOError, json.JSONDecodeError):
            # Failed to read or parse the metadata file; skip persistence
            pass

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

        for track_id, track in self.tracks.items():
            top, right, bottom, left = track["bbox"]
            base_label = track.get("label")
            display_label = track.get("display_name") or base_label or "loading..."
            color = (0, 255, 0) if base_label not in (None, "Unknown") else (0, 0, 255)

            # Hover highlight in Label Mode
            is_hovered = self.label_mode and track_id == self.hovered_track_id
            thickness = 4 if is_hovered else 2
            if is_hovered:
                color = (255, 255, 0)  # Yellow highlight in BGR

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
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
