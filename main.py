# main.py
import tkinter as tk

from face_recognition_app import FaceRecognitionApp


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root, camera_index=1)
    root.mainloop()


if __name__ == "__main__":
    main()
