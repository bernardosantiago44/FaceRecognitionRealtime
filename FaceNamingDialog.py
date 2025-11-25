# FaceNamingDialog.py
# Modal dialog for naming a face

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional


class FaceNamingDialog(tk.Toplevel):
    """
    Modal dialog for assigning or updating a person's name.
    
    Displays:
    - Face thumbnail (best available crop)
    - Read-only ID (e.g., persona_0231)
    - Text input for Name
    - Buttons: Save, Cancel, Clear name
    
    Returns result dict with action and name, or None if cancelled.
    """
    
    def __init__(
        self,
        parent,
        identity_id: str,
        face_thumbnail: Optional[np.ndarray],
        current_name: Optional[str] = None
    ):
        super().__init__(parent)
        self.title("Name Face")
        self.result = None
        
        self.identity_id = identity_id
        self.current_name = current_name
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Prevent resizing
        self.resizable(False, False)
        
        # Main container
        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Face thumbnail
        thumbnail_frame = tk.Frame(main_frame)
        thumbnail_frame.pack(pady=(0, 10))
        
        self.thumbnail_label = tk.Label(thumbnail_frame)
        self.thumbnail_label.pack()
        self._set_thumbnail(face_thumbnail)
        
        # ID field (read-only)
        id_frame = tk.Frame(main_frame)
        id_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(id_frame, text="ID:", width=8, anchor="w").pack(side=tk.LEFT)
        id_entry = tk.Entry(id_frame, width=25, state="readonly")
        id_entry.pack(side=tk.LEFT, padx=(0, 5))
        # Insert ID into readonly entry
        id_entry.configure(state="normal")
        id_entry.insert(0, identity_id)
        id_entry.configure(state="readonly")
        
        # Name field
        name_frame = tk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(name_frame, text="Name:", width=8, anchor="w").pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value=current_name or "")
        self.name_entry = tk.Entry(name_frame, width=25, textvariable=self.name_var)
        self.name_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Validation message
        self.validation_label = tk.Label(main_frame, text="", fg="red", height=1)
        self.validation_label.pack(fill=tk.X, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left-aligned Clear button
        self.clear_btn = tk.Button(
            btn_frame, 
            text="Clear Name", 
            command=self._on_clear,
            width=10
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Right-aligned Cancel and Save buttons
        self.save_btn = tk.Button(
            btn_frame, 
            text="Save", 
            command=self._on_save,
            width=8
        )
        self.save_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.cancel_btn = tk.Button(
            btn_frame, 
            text="Cancel", 
            command=self._on_cancel,
            width=8
        )
        self.cancel_btn.pack(side=tk.RIGHT)
        
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Bind Enter key to save
        self.name_entry.bind("<Return>", lambda e: self._on_save())
        
        # Bind Escape key to cancel
        self.bind("<Escape>", lambda e: self._on_cancel())
        
        # Focus on name entry
        self.name_entry.focus_set()
        self.name_entry.select_range(0, tk.END)
        
        # Center over parent
        self.update_idletasks()
        if parent is not None:
            x = parent.winfo_rootx() + parent.winfo_width() // 2 - self.winfo_width() // 2
            y = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
            self.geometry(f"+{x}+{y}")
    
    def _set_thumbnail(self, face_thumbnail: Optional[np.ndarray]):
        """Set the face thumbnail image."""
        if face_thumbnail is None or face_thumbnail.size == 0:
            # Show placeholder
            self.thumbnail_label.configure(
                text="No thumbnail\navailable",
                width=15,
                height=5,
                bg="#e0e0e0"
            )
            return
        
        try:
            # Convert BGR to RGB
            if len(face_thumbnail.shape) == 3 and face_thumbnail.shape[2] == 3:
                thumbnail_rgb = cv2.cvtColor(face_thumbnail, cv2.COLOR_BGR2RGB)
            else:
                thumbnail_rgb = face_thumbnail
            
            # Resize to reasonable display size (max 150px width or height)
            h, w = thumbnail_rgb.shape[:2]
            max_size = 150
            scale = min(max_size / w, max_size / h, 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if scale < 1.0:
                thumbnail_rgb = cv2.resize(
                    thumbnail_rgb, 
                    (new_w, new_h), 
                    interpolation=cv2.INTER_AREA
                )
            
            # Convert to PIL Image and then to PhotoImage
            img_pil = Image.fromarray(thumbnail_rgb)
            self.photo_image = ImageTk.PhotoImage(image=img_pil)
            
            self.thumbnail_label.configure(
                image=self.photo_image,
                text="",
                width=new_w,
                height=new_h
            )
        except Exception:
            # Fallback to placeholder
            self.thumbnail_label.configure(
                text="No thumbnail\navailable",
                width=15,
                height=5,
                bg="#e0e0e0"
            )
    
    def _validate_name(self, name: str) -> bool:
        """Validate the name input."""
        trimmed = name.strip()
        if not trimmed:
            self.validation_label.configure(text="Name cannot be empty")
            return False
        self.validation_label.configure(text="")
        return True
    
    def _on_save(self):
        """Handle Save button click."""
        name = self.name_var.get()
        
        if not self._validate_name(name):
            return
        
        self.result = {
            "action": "save",
            "name": name.strip()
        }
        self.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.destroy()
    
    def _on_clear(self):
        """Handle Clear Name button click."""
        self.result = {
            "action": "clear"
        }
        self.destroy()
