# unknown_tracker.py

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackSample:
    image: Optional[np.ndarray]
    blur: float
    area: int


@dataclass
class Track:
    track_id: int
    last_bbox: Tuple[int, int, int, int]
    embeddings: List[np.ndarray] = field(default_factory=list)
    samples: List[TrackSample] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    active: bool = True
    emitted: bool = False


@dataclass
class RegistrationCandidate:
    embedding: np.ndarray
    samples: List[Dict]
    meta: Dict


class UnknownTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_frames: int = 5,
        max_inactive_frames: int = 15,
        max_images: int = 3,
        blur_threshold: float = 100.0,
    ):
        self.iou_threshold = iou_threshold
        self.min_frames = min_frames
        self.max_inactive_frames = max_inactive_frames
        self.max_images = max_images
        self.blur_threshold = blur_threshold

        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 1

    @staticmethod
    def _iou(boxA, boxB) -> float:
        (xA1, yA1, xA2, yA2) = boxA
        (xB1, yB1, xB2, yB2) = boxB
        inter_x1 = max(xA1, xB1)
        inter_y1 = max(yA1, yB1)
        inter_x2 = min(xA2, xB2)
        inter_y2 = min(yA2, yB2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        boxA_area = (xA2 - xA1) * (yA2 - yA1)
        boxB_area = (xB2 - xB1) * (yB2 - yB1)
        union = boxA_area + boxB_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
        l, t, r, b = bbox
        return max(0, r - l) * max(0, b - t)

    def _create_track(
        self,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
        embedding: np.ndarray,
        face_image_bgr: Optional[np.ndarray],
        blur: float,
    ):
        area = self._bbox_area(bbox)
        sample = TrackSample(face_image_bgr, blur, area)
        t = Track(
            track_id=self._next_track_id,
            last_bbox=bbox,
            embeddings=[embedding],
            samples=[sample],
            first_frame=frame_idx,
            last_frame=frame_idx,
            active=True,
            emitted=False,
        )
        self._tracks[t.track_id] = t
        logger.debug(f"UnknownTracker: created track {t.track_id} at frame {frame_idx}")
        self._next_track_id += 1

    def _build_candidate(self, track: Track) -> Optional[RegistrationCandidate]:
        if track.emitted:
            return None

        life = track.last_frame - track.first_frame + 1
        if life < self.min_frames or len(track.embeddings) < self.min_frames:
            logger.debug(f"UnknownTracker: track {track.track_id} not ready (life={life}, embeddings={len(track.embeddings)}, min_frames={self.min_frames})")
            return None

        mean_emb = np.mean(track.embeddings, axis=0).astype("float32")

        # Convert samples to dicts and sort by blur, then area
        sample_dicts: List[Dict] = []
        for s in track.samples:
            if s.image is None:
                continue
            sample_dicts.append(
                {"image": s.image, "blur": float(s.blur), "area": int(s.area)}
            )

        sample_dicts.sort(key=lambda x: (x["blur"], x["area"]), reverse=True)
        if sample_dicts:
            selected = sample_dicts[: self.max_images]
        else:
            selected = []

        meta = {
            "source": "online_capture",
            "track_id": track.track_id,
            "first_frame": track.first_frame,
            "last_frame": track.last_frame,
            "num_observations": len(track.embeddings),
            "blur_threshold": self.blur_threshold,
        }

        track.emitted = True

        logger.info(f"UnknownTracker: built registration candidate from track {track.track_id} (embeddings={len(track.embeddings)}, samples={len(selected)})")
        return RegistrationCandidate(
            embedding=mean_emb,
            samples=selected,
            meta=meta,
        )

    def update(
        self,
        frame_idx: int,
        unknown_faces: List[Dict],
    ) -> List[RegistrationCandidate]:
        """
        unknown_faces: list of dicts:
          - bbox: (l,t,r,b)
          - embedding: np.ndarray
          - blur: float
          - face_image_bgr: np.ndarray
        """
        # Associate detections
        for face in unknown_faces:
            bbox = face["bbox"]
            emb = face["embedding"]
            img = face.get("face_image_bgr")
            blur = float(face.get("blur", 0.0))

            best_track_id = None
            best_iou = 0.0

            # First try IoU-based matching
            for track_id, track in self._tracks.items():
                if not track.active:
                    continue
                iou = self._iou(track.last_bbox, bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            # If IoU matching failed, try embedding-based matching as fallback
            # This handles cases where the face moves significantly between frames
            if best_track_id is None and emb is not None:
                best_emb_dist = float("inf")
                emb_threshold = 0.6  # Embedding distance threshold for same person
                for track_id, track in self._tracks.items():
                    if not track.active:
                        continue
                    if not track.embeddings:
                        continue
                    # Compare with mean embedding of the track
                    track_mean_emb = np.mean(track.embeddings, axis=0)
                    dist = float(np.linalg.norm(emb - track_mean_emb))
                    if dist < best_emb_dist and dist < emb_threshold:
                        best_emb_dist = dist
                        best_track_id = track_id
                if best_track_id is not None:
                    logger.debug(f"UnknownTracker: embedding fallback matched face to track {best_track_id} (dist={best_emb_dist:.3f})")

            if best_track_id is not None:
                track = self._tracks[best_track_id]
                track.last_bbox = bbox
                track.embeddings.append(emb)
                area = self._bbox_area(bbox)
                track.samples.append(TrackSample(img, blur, area))
                track.last_frame = frame_idx
                logger.debug(f"UnknownTracker: updated track {best_track_id} at frame {frame_idx} (embeddings={len(track.embeddings)})")
            else:
                self._create_track(frame_idx, bbox, emb, img, blur)

        # Deactivate stale
        for track in self._tracks.values():
            if track.active and frame_idx - track.last_frame > self.max_inactive_frames:
                logger.debug(f"UnknownTracker: deactivating stale track {track.track_id} (last_frame={track.last_frame}, current={frame_idx})")
                track.active = False

        candidates: List[RegistrationCandidate] = []

        for track in self._tracks.values():
            if track.emitted:
                continue

            if not track.active:
                cand = self._build_candidate(track)
                if cand:
                    candidates.append(cand)
            else:
                # Early registration for long-lived unknowns
                life = track.last_frame - track.first_frame + 1
                if life >= self.min_frames * 2:
                    cand = self._build_candidate(track)
                    if cand:
                        candidates.append(cand)

        return candidates
