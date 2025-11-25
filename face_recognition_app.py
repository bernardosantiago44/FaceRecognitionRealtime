# face_recognition_app.py
# Main application class for face recognition

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
from matching import find_best_match
from camera_utils import list_available_cameras
from config import T_KNOWN, BLUR_THRESHOLD, BLUR_MARGIN_UPGRADE, SCALE

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from CameraSelectionDialog import CameraSelectionDialog
from FaceNamingDialog import FaceNamingDialog


class FaceRecognitionApp:
    def __init__(self, root, camera_index: int = 0):
        self.root = root
        self.root.title("Face Recognition")

        # --- Label Mode state (session-only, not persisted) ---
        self.label_mode = False
        # Store current frame's face bounding boxes for click detection
        # Each entry: {"bbox": (left, top, right, bottom), "identity_id": str or None, "face_crop": np.ndarray}
        self.current_face_boxes = []
        # Store current frame for thumbnail retrieval
        self.current_frame = None

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

        # Bind click handler for Label Mode
        self.video_label.bind("<Button-1>", self._on_video_click)

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Label Mode toggle button
        self.label_mode_button = tk.Button(
            self.bottom_frame,
            text="Label Mode: Off",
            command=self._toggle_label_mode,
            width=15
        )
        self.label_mode_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.select_cam_button = tk.Button(
            self.bottom_frame,
            text="Select Camera",
            command=self.change_camera
        )
        self.select_cam_button.pack(side=tk.LEFT, padx=5, pady=5)

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

    def _toggle_label_mode(self):
        """Toggle Label Mode on/off."""
        self.label_mode = not self.label_mode
        if self.label_mode:
            self.label_mode_button.config(
                text="Label Mode: On",
                relief=tk.SUNKEN,
                bg="#90EE90"  # Light green background when active
            )
        else:
            self.label_mode_button.config(
                text="Label Mode: Off",
                relief=tk.RAISED,
                bg=self.root.cget("bg")  # Default background
            )

    def _on_video_click(self, event):
        """Handle click on video feed. In Label Mode, check if a face was clicked."""
        if not self.label_mode:
            return

        # Get click coordinates relative to the video label widget
        click_x = event.x
        click_y = event.y

        # Calculate offset if image is centered within the label
        # The label might be larger than the image due to fill=BOTH, expand=True
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        # Get actual image dimensions from the stored frame
        if self.current_frame is not None:
            img_height, img_width = self.current_frame.shape[:2]
        else:
            # Fallback to camera settings
            img_width = 640
            img_height = 480

        # Calculate offset (image is centered in label)
        offset_x = max(0, (label_width - img_width) // 2)
        offset_y = max(0, (label_height - img_height) // 2)

        # Adjust click coordinates to image space
        adjusted_x = click_x - offset_x
        adjusted_y = click_y - offset_y

        # Check if click is within image bounds
        if adjusted_x < 0 or adjusted_x >= img_width or adjusted_y < 0 or adjusted_y >= img_height:
            return

        # Make a copy of face boxes to avoid race conditions during frame updates
        face_boxes = list(self.current_face_boxes)

        # Check if click falls within any detected face bounding box
        for face_info in face_boxes:
            left, top, right, bottom = face_info["bbox"]
            if left <= adjusted_x <= right and top <= adjusted_y <= bottom:
                # Face was clicked - trigger naming dialog
                identity_id = face_info.get("identity_id")
                face_crop = face_info.get("face_crop")
                self._on_face_clicked(identity_id, bbox=face_info["bbox"], face_crop=face_crop)
                return

    def _on_face_clicked(self, identity_id, bbox, face_crop=None):
        """
        Called when a face is clicked in Label Mode.
        Opens the naming dialog for the selected face.
        """
        # Check if the face is unknown/not registered
        if identity_id is None:
            messagebox.showinfo(
                "Unknown Face",
                "This face is not yet registered in the system.\n"
                "Please wait for automatic registration to complete."
            )
            return

        # Get current name from metadata
        meta = self.index.get_meta(identity_id)
        current_name = meta.get("name") if meta else None

        # Open naming dialog
        dialog = FaceNamingDialog(
            self.root,
            identity_id=identity_id,
            face_thumbnail=face_crop,
            current_name=current_name
        )
        # Wait for dialog to close
        self.root.wait_window(dialog)

        result = dialog.result
        if result is None:
            return  # User cancelled

        if result["action"] == "save":
            # Update name in registry and persist to disk
            new_name = result["name"]
            self._update_person_name(identity_id, new_name)
        elif result["action"] == "clear":
            # Clear name (revert to showing ID)
            self._update_person_name(identity_id, None)

    def _update_person_name(self, identity_id: str, name):
        """Update a person's name in the registry and persist to disk."""
        # Update in-memory index
        self.index.update_name(identity_id, name)
        # Persist to disk
        self.store.update_name(identity_id, name)


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

        # Store current frame for thumbnail retrieval
        self.current_frame = frame.copy()

        # Clear face boxes for this frame (for Label Mode click detection)
        self.current_face_boxes = []

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
            current_identity_id = None  # Track identity for Label Mode

            if known_embs.size > 0:
                idx_match, dist = find_best_match(emb, known_embs)
                if idx_match is not None and dist is not None and dist <= T_KNOWN:
                    identity_id = ids[idx_match]
                    current_identity_id = identity_id
                    # Use name from metadata if available, otherwise use ID
                    meta = self.index.get_meta(identity_id)
                    person_name = meta.get("name") if meta else None
                    label = person_name if person_name else identity_id
                    color = (0, 255, 0)

                    # upgrade path if this identity was low-quality
                    quality = meta.get("quality", {}) if meta else {}
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

            # Store face box for Label Mode click detection (with face crop for thumbnail)
            self.current_face_boxes.append({
                "bbox": (left, top, right, bottom),
                "identity_id": current_identity_id,
                "face_crop": face_crop.copy() if face_crop.size > 0 else None
            })

            # draw
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            # Add visual cue when Label Mode is On (highlight border)
            if self.label_mode:
                # Draw a yellow border to indicate clickable face
                cv2.rectangle(frame, (left - 2, top - 2), (right + 2, bottom + 2), (255, 255, 0), 2)

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
