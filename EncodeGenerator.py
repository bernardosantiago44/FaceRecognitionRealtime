# EncodeGenerator.py

import os
import cv2
import face_recognition
import numpy as np
from identity_store import IdentityStore


FOLDER_PATH = "Images"


def encode_image(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img_rgb)
    if not boxes:
        return None

    # For bootstrap, take the first detected face
    encodings = face_recognition.face_encodings(img_rgb, boxes)
    if not encodings:
        return None

    return np.array(encodings[0], dtype="float32")


def main():
    store = IdentityStore(root_dir="identities")

    path_list = os.listdir(FOLDER_PATH)

    print("Bootstrapping identities from Images/")

    created = 0
    for filename in path_list:
        if filename.startswith("."):
            continue

        img_path = os.path.join(FOLDER_PATH, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        embedding = encode_image(img)
        if embedding is None:
            print(f"No face found in {img_path}, skipped")
            continue

        # Use filename (without extension) as initial human label
        label = os.path.splitext(filename)[0]
        identity_id = store.create_identity(
            embedding=embedding,
            face_image_bgr=img,
            source="bootstrap",
            label=label,
        )

        print(f"Created identity {identity_id} from {filename}")
        created += 1

    print(f"Bootstrap complete. {created} identities created.")


if __name__ == "__main__":
    main()
