import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.clip_builder import ClipBuilder


class FakeVideoWriter:
    def __init__(self):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        pass


class FakeCv2:
    @staticmethod
    def VideoWriter_fourcc(*codec):
        return 0

    @staticmethod
    def VideoWriter(path, fourcc, fps, size):
        return FakeVideoWriter()


class ClipBuilderTest(unittest.TestCase):
    def test_save_candidate_clip_allows_missing_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ClipBuilder(output_dir=temp_dir)
            candidate = {
                "camera_id": "cam1",
                "candidate_id": "c1",
                "timestamp_ms": 1000,
            }
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frames = [
                {
                    "camera_id": "cam1",
                    "frame_id": 1,
                    "timestamp_ms": 1000,
                    "frame": frame,
                    "width": 16,
                    "height": 16,
                    "fps": 5,
                    "source_uri": "video.mp4",
                },
                {
                    "camera_id": "cam1",
                    "frame_id": 2,
                    "timestamp_ms": 1200,
                    "frame": frame,
                    "width": 16,
                    "height": 16,
                    "fps": 5,
                    "source_uri": "video.mp4",
                },
            ]

            with patch("services.clip_builder.cv2", FakeCv2):
                saved = builder.save_event_clip(
                    candidate=candidate,
                    verification=None,
                    frame_packets=frames,
                )

            metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["category"], "candidates")
            self.assertIn("candidates", saved["clip_path"])
            self.assertIsNone(metadata["verification"])


if __name__ == "__main__":
    unittest.main()
