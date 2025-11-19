import tkinter as tk

class CameraSelectionDialog(tk.Toplevel):
    def __init__(self, parent, camera_indices, current_index=None):
        super().__init__(parent)
        self.title("Select Camera")
        self.result = None

        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="Choose a camera:").pack(padx=10, pady=(10, 5))

        # Map labels ("Camera 0") to indices
        options = [f"Camera {idx}" for idx in camera_indices]
        self._label_to_idx = {label: idx for label, idx in zip(options, camera_indices)}

        default_label = (
            f"Camera {current_index}"
            if current_index in camera_indices
            else options[0]
        )

        self.var = tk.StringVar(value=default_label)
        option_menu = tk.OptionMenu(self, self.var, *options)
        option_menu.pack(padx=10, pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=(5, 10))

        tk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(
            side=tk.RIGHT, padx=5
        )
        tk.Button(btn_frame, text="OK", command=self._on_ok).pack(
            side=tk.RIGHT, padx=5
        )

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Center over parent (optional)
        self.update_idletasks()
        if parent is not None:
            x = parent.winfo_rootx() + parent.winfo_width() // 2 - self.winfo_width() // 2
            y = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
            self.geometry(f"+{x}+{y}")

    def _on_ok(self):
        label = self.var.get()
        self.result = self._label_to_idx[label]
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()
