# test_registration_worker.py
"""Tests for the RegistrationWorker class."""

import os
import shutil
import tempfile
import queue
import threading
import time
import unittest
import numpy as np

from identity_store import IdentityStore
from in_memory_index import InMemoryIndex
from unknown_tracker import RegistrationCandidate
from registration_worker import RegistrationWorker, RegistrationJob


class TestRegistrationWorker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with a temporary identities directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = IdentityStore(root_dir=self.temp_dir)
        self.index = InMemoryIndex(self.store)
        self.job_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.worker = RegistrationWorker(
            job_queue=self.job_queue,
            store=self.store,
            index=self.index,
            t_known=0.5,
            margin=0.05,
            blur_threshold=100.0,
            max_images=3,
            stop_event=self.stop_event,
        )
        self.worker.start()
    
    def tearDown(self):
        """Clean up temporary directory and stop worker."""
        self.worker.stop()
        self.worker.join(timeout=2)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_fake_embedding(self, seed=0):
        """Create a fake 128-dimensional embedding for testing."""
        np.random.seed(seed)
        return np.random.rand(128).astype("float32")
    
    def _create_fake_image(self):
        """Create a fake BGR image for testing."""
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def _create_candidate(self, embedding, track_id=1):
        """Create a RegistrationCandidate for testing."""
        samples = [
            {"image": self._create_fake_image(), "blur": 150.0, "area": 10000},
            {"image": self._create_fake_image(), "blur": 140.0, "area": 9000},
        ]
        meta = {
            "source": "online_capture",
            "track_id": track_id,
            "first_frame": 1,
            "last_frame": 10,
            "num_observations": 5,
            "blur_threshold": 100.0,
        }
        return RegistrationCandidate(embedding=embedding, samples=samples, meta=meta)
    
    def test_new_identity_registration(self):
        """Test that a new identity is created for unknown embedding."""
        embedding = self._create_fake_embedding(seed=1)
        candidate = self._create_candidate(embedding, track_id=1)
        
        job = RegistrationJob(candidate=candidate)
        self.job_queue.put(job)
        
        # Wait for processing
        self.job_queue.join()
        time.sleep(0.1)  # Brief wait for index update
        
        # Check that identity was created
        ids, _ = self.index.get_snapshot()
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0].startswith("person_"))
        
        # Verify files were created
        identity_dir = os.path.join(self.temp_dir, ids[0])
        self.assertTrue(os.path.exists(identity_dir))
        self.assertTrue(os.path.exists(os.path.join(identity_dir, "meta.json")))
        self.assertTrue(os.path.exists(os.path.join(identity_dir, "embeddings.npz")))
    
    def test_duplicate_registration_blocked(self):
        """Test that duplicate registrations are blocked for similar embeddings."""
        embedding1 = self._create_fake_embedding(seed=2)
        candidate1 = self._create_candidate(embedding1, track_id=1)
        
        job1 = RegistrationJob(candidate=candidate1)
        self.job_queue.put(job1)
        self.job_queue.join()
        time.sleep(0.1)
        
        # Try to register same embedding again
        candidate2 = self._create_candidate(embedding1, track_id=2)  # Same embedding
        job2 = RegistrationJob(candidate=candidate2)
        self.job_queue.put(job2)
        self.job_queue.join()
        time.sleep(0.1)
        
        # Should still have only 1 identity (duplicate blocked)
        ids, _ = self.index.get_snapshot()
        self.assertEqual(len(ids), 1)
    
    def test_different_persons_registered_separately(self):
        """Test that different persons get separate identities."""
        embedding1 = self._create_fake_embedding(seed=10)
        embedding2 = self._create_fake_embedding(seed=20)
        
        candidate1 = self._create_candidate(embedding1, track_id=1)
        candidate2 = self._create_candidate(embedding2, track_id=2)
        
        job1 = RegistrationJob(candidate=candidate1)
        job2 = RegistrationJob(candidate=candidate2)
        
        self.job_queue.put(job1)
        self.job_queue.put(job2)
        self.job_queue.join()
        time.sleep(0.2)
        
        # Should have 2 identities
        ids, _ = self.index.get_snapshot()
        self.assertEqual(len(ids), 2)


if __name__ == "__main__":
    unittest.main()
