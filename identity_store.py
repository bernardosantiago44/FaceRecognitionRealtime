# identity_store.py

import os
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from ResourcePath import resource_path


class IdentityStore:
    def __init__(self, root_dir: str = "identities"):
        self.root_dir = resource_path(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    # ---------- Read side ----------

    def load_all_embeddings_and_meta(self) -> Tuple[List[str], np.ndarray, Dict[str, dict]]:
        """
        Load one representative embedding and meta per identity.

        Representative embedding: mean of embeddings.npz["embeddings"].
        """
        ids: List[str] = []
        emb_list: List[np.ndarray] = []
        meta_map: Dict[str, dict] = {}

        for ident in sorted(os.listdir(self.root_dir)):
            ident_dir = resource_path(os.path.join(self.root_dir, ident))
            if not os.path.isdir(ident_dir):
                continue

            emb_path = resource_path(os.path.join(ident_dir, "embeddings.npz"))
            if not os.path.exists(emb_path):
                continue

            try:
                data = np.load(emb_path)
                embs = data.get("embeddings")
                if embs is None or len(embs) == 0:
                    continue
                rep = embs.mean(axis=0).astype("float32")
            except Exception:
                continue

            meta_path = resource_path(os.path.join(ident_dir, "meta.json"))
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

            ids.append(ident)
            emb_list.append(rep)
            meta_map[ident] = meta

        if not emb_list:
            return [], np.array([]), {}

        embeddings = np.vstack(emb_list).astype("float32")
        return ids, embeddings, meta_map

    def load_meta(self, identity_id: str) -> Optional[dict]:
        ident_dir = resource_path(os.path.join(self.root_dir, identity_id))
        meta_path = resource_path(os.path.join(ident_dir, "meta.json"))
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    # ---------- Write side ----------

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _generate_identity_id(self) -> str:
        existing = [
            name
            for name in os.listdir(self.root_dir)
            if os.path.isdir(resource_path(os.path.join(self.root_dir, name)))
            and name.startswith("person_")
        ]
        max_num = 0
        for name in existing:
            try:
                num = int(name.split("_")[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                continue
        return f"person_{max_num + 1:06d}"

    def create_identity_from_samples(
        self,
        embedding: np.ndarray,
        samples: List[Dict],
        source: str,
        label: Optional[str],
        extra_meta: Optional[dict],
        max_images: int,
    ) -> str:
        """
        samples: list of dicts:
            {
              "image": np.ndarray (BGR),
              "blur": float,
              "area": int
            }
        """
        if embedding.ndim != 1:
            raise ValueError("embedding must be 1D")

        identity_id = self._generate_identity_id()
        ident_dir = resource_path(os.path.join(self.root_dir, identity_id))
        os.makedirs(ident_dir, exist_ok=False)

        # Save embeddings: for now just the representative embedding
        emb_path = resource_path(os.path.join(ident_dir, "embeddings.npz"))
        np.savez_compressed(emb_path, embeddings=embedding.reshape(1, -1).astype("float32"))

        # Sort samples by blur desc, then area desc
        valid_samples = [s for s in samples if s.get("image") is not None]
        valid_samples.sort(key=lambda s: (s.get("blur", 0.0), s.get("area", 0)), reverse=True)

        num_images = 0
        blur_values = [s.get("blur", 0.0) for s in valid_samples]

        if valid_samples:
            faces_dir = resource_path(os.path.join(ident_dir, "faces"))
            os.makedirs(faces_dir, exist_ok=True)
            for i, s in enumerate(valid_samples[:max_images], start=1):
                img = s["image"]
                img_name = f"face_{i:03d}.jpg"
                img_path = resource_path(os.path.join(faces_dir, img_name))
                cv2.imwrite(img_path, img)
                num_images += 1

        now = self._now_iso()

        if blur_values:
            max_blur = float(max(blur_values))
            min_blur = float(min(blur_values))
            mean_blur = float(sum(blur_values) / len(blur_values))
        else:
            max_blur = min_blur = mean_blur = 0.0

        BLUR_THRESHOLD = extra_meta.get("blur_threshold", 100.0) if extra_meta else 100.0
        needs_refresh = bool(max_blur < BLUR_THRESHOLD)

        quality = {
            "max_blur": max_blur,
            "min_blur": min_blur,
            "mean_blur": mean_blur,
            "needs_refresh": needs_refresh,
        }

        meta = {
            "id": identity_id,
            "created_at": now,
            "last_updated_at": now,
            "label": label,
            "source": source,
            "num_embeddings": 1,
            "num_images": num_images,
            "quality": quality,
        }
        if extra_meta:
            meta.update({k: v for k, v in extra_meta.items() if k not in meta})

        meta_path = resource_path(os.path.join(ident_dir, "meta.json"))
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return identity_id

    def update_identity_with_sample(
        self,
        identity_id: str,
        new_embedding: np.ndarray,
        sample: Dict,
        max_images: int,
        blur_threshold: float,
    ) -> Optional[dict]:
        """
        Append new embedding and possibly replace one of the stored images with a better one.
        Returns updated meta dict or None on failure.
        """
        ident_dir = resource_path(os.path.join(self.root_dir, identity_id))
        emb_path = resource_path(os.path.join(ident_dir, "embeddings.npz"))
        meta_path = resource_path(os.path.join(ident_dir, "meta.json"))

        if not os.path.exists(ident_dir) or not os.path.exists(emb_path):
            return None

        try:
            data = np.load(emb_path)
            embs = data.get("embeddings")
            if embs is None or embs.size == 0:
                embs = new_embedding.reshape(1, -1).astype("float32")
            else:
                embs = np.vstack([embs, new_embedding.reshape(1, -1)]).astype("float32")
            np.savez_compressed(emb_path, embeddings=embs)
        except Exception:
            return None

        # Handle images
        img = sample.get("image")
        blur = float(sample.get("blur", 0.0))
        area = int(sample.get("area", 0))

        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {}

            faces_dir = resource_path(os.path.join(ident_dir, "faces"))
            os.makedirs(faces_dir, exist_ok=True)

            # Load existing images blur stats from meta if present
            quality = meta.get("quality", {})
            max_blur_prev = float(quality.get("max_blur", 0.0))

            # If new image is sharper than previous max, write or replace one file
            if img is not None and blur > max_blur_prev:
                # Save as next index or overwrite lowest blur slot
                existing = sorted(
                    [
                        f
                        for f in os.listdir(faces_dir)
                        if f.lower().startswith("face_") and f.lower().endswith(".jpg")
                    ]
                )
                if len(existing) < max_images:
                    img_name = f"face_{len(existing)+1:03d}.jpg"
                else:
                    # Overwrite first for simplicity
                    img_name = existing[0]
                cv2.imwrite(resource_path(os.path.join(faces_dir, img_name)), img)

            # Recompute quality fields
            # For simplicity: update using max(old, new)
            max_blur = max(max_blur_prev, blur)
            # Do not overfit mean/min here for demo
            quality["max_blur"] = max_blur
            quality["needs_refresh"] = bool(max_blur < blur_threshold)

            meta["last_updated_at"] = self._now_iso()
            meta["quality"] = quality

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            return meta
        except Exception:
            return None
