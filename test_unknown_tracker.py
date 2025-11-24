# test_unknown_tracker.py
"""Tests for the UnknownTracker class to verify registration pipeline."""

import numpy as np
import unittest
from unknown_tracker import UnknownTracker, RegistrationCandidate


class TestUnknownTracker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = UnknownTracker(
            iou_threshold=0.3,
            min_frames=5,
            max_inactive_frames=15,
            max_images=3,
            blur_threshold=100.0,
        )
    
    def _create_fake_embedding(self, seed=0):
        """Create a fake 128-dimensional embedding for testing."""
        np.random.seed(seed)
        return np.random.rand(128).astype("float32")
    
    def _create_unknown_face(self, bbox, embedding, blur=100.0):
        """Create an unknown face dict for testing."""
        # Create a small fake image (just for testing, not real)
        fake_image = np.zeros((50, 50, 3), dtype=np.uint8)
        return {
            "bbox": bbox,
            "embedding": embedding,
            "face_image_bgr": fake_image,
            "blur": blur,
        }
    
    def test_track_creation(self):
        """Test that new tracks are created for unknown faces."""
        bbox = (100, 100, 200, 200)  # (left, top, right, bottom)
        embedding = self._create_fake_embedding(seed=1)
        unknown_faces = [self._create_unknown_face(bbox, embedding)]
        
        candidates = self.tracker.update(frame_idx=1, unknown_faces=unknown_faces)
        
        # Should not have candidates yet (needs min_frames=5)
        self.assertEqual(len(candidates), 0)
        # Should have created 1 track
        self.assertEqual(len(self.tracker._tracks), 1)
    
    def test_track_iou_matching(self):
        """Test that faces are matched to existing tracks via IoU."""
        bbox1 = (100, 100, 200, 200)
        bbox2 = (105, 105, 205, 205)  # Overlapping with bbox1
        embedding = self._create_fake_embedding(seed=2)
        
        # Frame 1: Create track
        self.tracker.update(frame_idx=1, unknown_faces=[
            self._create_unknown_face(bbox1, embedding)
        ])
        
        # Frame 2: Should match to existing track via IoU
        self.tracker.update(frame_idx=2, unknown_faces=[
            self._create_unknown_face(bbox2, embedding)
        ])
        
        # Should still have only 1 track (matched via IoU)
        self.assertEqual(len(self.tracker._tracks), 1)
        
        # Track should have 2 embeddings
        track = list(self.tracker._tracks.values())[0]
        self.assertEqual(len(track.embeddings), 2)
    
    def test_track_embedding_fallback_matching(self):
        """Test that embedding-based fallback matching works when IoU fails."""
        bbox1 = (100, 100, 200, 200)
        bbox2 = (300, 300, 400, 400)  # Non-overlapping - IoU should fail
        embedding = self._create_fake_embedding(seed=3)  # Same person
        
        # Frame 1: Create track at bbox1
        self.tracker.update(frame_idx=1, unknown_faces=[
            self._create_unknown_face(bbox1, embedding)
        ])
        
        # Frame 2: Face moved to bbox2 (no IoU overlap)
        # Should still match via embedding similarity
        self.tracker.update(frame_idx=2, unknown_faces=[
            self._create_unknown_face(bbox2, embedding)
        ])
        
        # Should still have only 1 track (matched via embedding)
        self.assertEqual(len(self.tracker._tracks), 1)
        
        # Track should have 2 embeddings
        track = list(self.tracker._tracks.values())[0]
        self.assertEqual(len(track.embeddings), 2)
    
    def test_different_persons_create_separate_tracks(self):
        """Test that different embeddings create separate tracks."""
        bbox1 = (100, 100, 200, 200)
        bbox2 = (300, 300, 400, 400)  # Non-overlapping
        embedding1 = self._create_fake_embedding(seed=10)
        embedding2 = self._create_fake_embedding(seed=20)  # Different person
        
        # Frame 1: Create track for person 1
        self.tracker.update(frame_idx=1, unknown_faces=[
            self._create_unknown_face(bbox1, embedding1)
        ])
        
        # Frame 2: Different person at bbox2
        self.tracker.update(frame_idx=2, unknown_faces=[
            self._create_unknown_face(bbox2, embedding2)
        ])
        
        # Should have 2 tracks (different embeddings)
        self.assertEqual(len(self.tracker._tracks), 2)
    
    def test_registration_candidate_after_min_frames(self):
        """Test that registration candidate is generated after min_frames."""
        bbox = (100, 100, 200, 200)
        embedding = self._create_fake_embedding(seed=4)
        
        # Update for min_frames * 2 = 10 frames to trigger early registration
        for i in range(1, 11):
            # Slight bbox variation to simulate real scenario
            varied_bbox = (100 + i, 100 + i, 200 + i, 200 + i)
            self.tracker.update(frame_idx=i, unknown_faces=[
                self._create_unknown_face(varied_bbox, embedding)
            ])
        
        # Check if track has accumulated enough embeddings
        track = list(self.tracker._tracks.values())[0]
        self.assertEqual(len(track.embeddings), 10)
        
        # Frame 10 should have triggered early registration (life >= min_frames * 2)
        # Actually, the candidate is returned on the frame that meets criteria
        # Since life=10 >= min_frames*2=10, and embeddings=10 >= min_frames=5
        # Let's check if a candidate was emitted by checking the emitted flag
        self.assertTrue(track.emitted)
    
    def test_multiple_unknown_faces_same_frame(self):
        """Test handling multiple unknown faces in the same frame."""
        embedding1 = self._create_fake_embedding(seed=100)
        embedding2 = self._create_fake_embedding(seed=200)
        
        unknown_faces = [
            self._create_unknown_face((100, 100, 200, 200), embedding1),
            self._create_unknown_face((300, 300, 400, 400), embedding2),
        ]
        
        self.tracker.update(frame_idx=1, unknown_faces=unknown_faces)
        
        # Should create 2 separate tracks
        self.assertEqual(len(self.tracker._tracks), 2)
    
    def test_track_deactivation_after_inactive_frames(self):
        """Test that tracks are deactivated after max_inactive_frames."""
        bbox = (100, 100, 200, 200)
        embedding = self._create_fake_embedding(seed=5)
        
        # Frame 1: Create track
        self.tracker.update(frame_idx=1, unknown_faces=[
            self._create_unknown_face(bbox, embedding)
        ])
        
        # Track should be active
        track = list(self.tracker._tracks.values())[0]
        self.assertTrue(track.active)
        
        # Skip many frames without seeing the face
        for i in range(2, 20):
            self.tracker.update(frame_idx=i, unknown_faces=[])
        
        # Track should be deactivated (not seen for > 15 frames)
        self.assertFalse(track.active)
    
    def test_no_duplicate_candidates(self):
        """Test that emitted tracks don't generate duplicate candidates."""
        bbox = (100, 100, 200, 200)
        embedding = self._create_fake_embedding(seed=6)
        
        # Accumulate enough embeddings
        all_candidates = []
        for i in range(1, 15):
            varied_bbox = (100 + i, 100 + i, 200 + i, 200 + i)
            candidates = self.tracker.update(frame_idx=i, unknown_faces=[
                self._create_unknown_face(varied_bbox, embedding)
            ])
            all_candidates.extend(candidates)
        
        # Should have generated exactly 1 candidate
        self.assertEqual(len(all_candidates), 1)
        
        # Track should be marked as emitted
        track = list(self.tracker._tracks.values())[0]
        self.assertTrue(track.emitted)


if __name__ == "__main__":
    unittest.main()
