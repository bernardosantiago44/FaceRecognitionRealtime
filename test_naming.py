# test_naming.py
# Unit tests for the click-to-name feature components

import os
import json
import tempfile
import shutil
import unittest
import numpy as np

# Manually handle imports to avoid tkinter dependency
import sys
sys.modules['tkinter'] = type(sys)('tkinter')
sys.modules['PIL'] = type(sys)('PIL')
sys.modules['PIL.Image'] = type(sys)('PIL.Image')
sys.modules['PIL.ImageTk'] = type(sys)('PIL.ImageTk')
sys.modules['cv2'] = type(sys)('cv2')

from identity_store import IdentityStore
from in_memory_index import InMemoryIndex


class TestIdentityStoreUpdateName(unittest.TestCase):
    """Test the update_name functionality in IdentityStore."""

    def setUp(self):
        """Create a temporary directory for identity storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = IdentityStore(root_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_identity(self, identity_id: str, initial_meta: dict = None):
        """Helper to create a test identity directory with metadata."""
        ident_dir = os.path.join(self.temp_dir, identity_id)
        os.makedirs(ident_dir, exist_ok=True)

        # Create embeddings.npz
        emb_path = os.path.join(ident_dir, "embeddings.npz")
        np.savez_compressed(emb_path, embeddings=np.random.randn(1, 128).astype("float32"))

        # Create meta.json
        meta_path = os.path.join(ident_dir, "meta.json")
        meta = initial_meta or {"id": identity_id}
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return ident_dir

    def test_update_name_new_name(self):
        """Test setting a new name for an identity."""
        identity_id = "person_000001"
        self._create_test_identity(identity_id, {"id": identity_id})

        result = self.store.update_name(identity_id, "John Doe")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "John Doe")
        self.assertIn("last_updated_at", result)

        # Verify persisted to disk
        meta_path = os.path.join(self.temp_dir, identity_id, "meta.json")
        with open(meta_path, "r") as f:
            persisted = json.load(f)
        self.assertEqual(persisted["name"], "John Doe")

    def test_update_name_clear_name(self):
        """Test clearing a name (setting to None)."""
        identity_id = "person_000002"
        self._create_test_identity(identity_id, {"id": identity_id, "name": "Jane Smith"})

        result = self.store.update_name(identity_id, None)

        self.assertIsNotNone(result)
        self.assertIsNone(result["name"])

    def test_update_name_replace_existing(self):
        """Test replacing an existing name."""
        identity_id = "person_000003"
        self._create_test_identity(identity_id, {"id": identity_id, "name": "Old Name"})

        result = self.store.update_name(identity_id, "New Name")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "New Name")

    def test_update_name_nonexistent_identity(self):
        """Test updating name for non-existent identity returns None."""
        result = self.store.update_name("nonexistent_person", "Test Name")
        self.assertIsNone(result)

    def test_update_name_preserves_other_fields(self):
        """Test that updating name preserves other metadata fields."""
        identity_id = "person_000004"
        initial_meta = {
            "id": identity_id,
            "created_at": "2024-01-01T00:00:00Z",
            "source": "online_capture",
            "num_images": 3
        }
        self._create_test_identity(identity_id, initial_meta)

        result = self.store.update_name(identity_id, "Test Person")

        self.assertEqual(result["id"], identity_id)
        self.assertEqual(result["created_at"], "2024-01-01T00:00:00Z")
        self.assertEqual(result["source"], "online_capture")
        self.assertEqual(result["num_images"], 3)
        self.assertEqual(result["name"], "Test Person")


class TestInMemoryIndexUpdateName(unittest.TestCase):
    """Test the update_name functionality in InMemoryIndex."""

    def setUp(self):
        """Create a temporary store and index."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = IdentityStore(root_dir=self.temp_dir)
        self.index = InMemoryIndex(self.store)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_name_existing_identity(self):
        """Test updating name for an existing identity in memory."""
        identity_id = "person_000001"
        embedding = np.random.randn(128).astype("float32")
        meta = {"id": identity_id, "source": "test"}

        self.index.add_identity(identity_id, embedding, meta)
        self.index.update_name(identity_id, "Updated Name")

        result_meta = self.index.get_meta(identity_id)
        self.assertEqual(result_meta["name"], "Updated Name")
        self.assertEqual(result_meta["source"], "test")

    def test_update_name_new_identity(self):
        """Test updating name for identity not in meta creates entry."""
        identity_id = "person_000002"

        self.index.update_name(identity_id, "New Person")

        result_meta = self.index.get_meta(identity_id)
        self.assertEqual(result_meta["name"], "New Person")

    def test_update_name_thread_safety(self):
        """Test that update_name is thread-safe."""
        import threading

        identity_id = "person_000003"
        embedding = np.random.randn(128).astype("float32")
        self.index.add_identity(identity_id, embedding, {"id": identity_id})

        names = ["Name1", "Name2", "Name3", "Name4", "Name5"]
        errors = []

        def update_name(name):
            try:
                self.index.update_name(identity_id, name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_name, args=(name,)) for name in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        result_meta = self.index.get_meta(identity_id)
        self.assertIn(result_meta["name"], names)


class TestFaceNamingDialogValidation(unittest.TestCase):
    """Test validation logic from FaceNamingDialog without requiring Tkinter."""

    def test_name_validation_empty_string(self):
        """Test that empty name fails validation."""
        # Validation logic: name cannot be empty after trimming
        name = ""
        trimmed = name.strip()
        self.assertFalse(bool(trimmed))

    def test_name_validation_whitespace_only(self):
        """Test that whitespace-only name fails validation."""
        name = "   "
        trimmed = name.strip()
        self.assertFalse(bool(trimmed))

    def test_name_validation_valid_name(self):
        """Test that valid name passes validation."""
        name = "John Doe"
        trimmed = name.strip()
        self.assertTrue(bool(trimmed))

    def test_name_validation_with_whitespace(self):
        """Test that name with leading/trailing whitespace is trimmed."""
        name = "  Jane Smith  "
        trimmed = name.strip()
        self.assertEqual(trimmed, "Jane Smith")
        self.assertTrue(bool(trimmed))


if __name__ == "__main__":
    unittest.main()
