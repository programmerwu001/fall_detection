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
    def test_save_debug_copy_true_writes_private_clip_and_debug_mirror(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private_dir = root / "private_events"
            debug_dir = root / "events"
            builder = ClipBuilder(
                output_dir=debug_dir,
                internal_output_dir=private_dir,
                save_debug_raw_event_copy=True,
            )
            candidate = {
                "camera_id": "cam1",
                "candidate_id": "c1",
                "timestamp_ms": 1000,
                "source_uri": "E:/raw/source.mp4",
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
                    "source_uri": "E:/raw/source.mp4",
                }
            ]

            def write_fake_clip(clip_path, frame_packets, fps, width, height):
                clip_path.write_bytes(b"private clip")
                return len(frame_packets)

            with patch.object(builder, "_write_video", side_effect=write_fake_clip):
                saved = builder.save_event_clip(
                    candidate=candidate,
                    verification=None,
                    frame_packets=frames,
                )

            private_clip = Path(saved["clip_path"])
            debug_clip = Path(saved["debug_clip_path"])
            debug_metadata = json.loads(
                Path(saved["debug_metadata_path"]).read_text(encoding="utf-8")
            )

            self.assertTrue(private_clip.exists())
            self.assertTrue(debug_clip.exists())
            self.assertTrue(private_clip.is_relative_to(private_dir))
            self.assertTrue(debug_clip.is_relative_to(debug_dir))
            self.assertNotEqual(private_clip, debug_clip)
            self.assertEqual(debug_clip.read_bytes(), b"private clip")
            serialized = json.dumps(debug_metadata, ensure_ascii=False)
            self.assertNotIn("source_uri", serialized)
            self.assertNotIn("clip_path", serialized)
            self.assertNotIn("source.mp4", serialized)

    def test_save_debug_copy_false_writes_only_private_clip_and_moves_old_debug_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private_dir = root / "private_events"
            debug_dir = root / "events"
            disabled_dir = root / "disabled_debug_events"
            old_dir = debug_dir / "cam1" / "20260627"
            old_dir.mkdir(parents=True)
            (old_dir / "event_1.mp4").write_bytes(b"old debug")
            (old_dir / "event_1.json").write_text(
                json.dumps({"clip_path": "E:/events/event_1.mp4"}),
                encoding="utf-8",
            )
            builder = ClipBuilder(
                output_dir=debug_dir,
                internal_output_dir=private_dir,
                disabled_debug_dir=disabled_dir,
                save_debug_raw_event_copy=False,
            )
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
                    "source_uri": "E:/raw/source.mp4",
                }
            ]

            def write_fake_clip(clip_path, frame_packets, fps, width, height):
                clip_path.write_bytes(b"private clip")
                return len(frame_packets)

            with patch.object(builder, "_write_video", side_effect=write_fake_clip):
                saved = builder.save_event_clip(
                    candidate=candidate,
                    verification=None,
                    frame_packets=frames,
                )

            self.assertTrue(Path(saved["clip_path"]).exists())
            self.assertNotIn("debug_clip_path", saved)
            self.assertFalse((old_dir / "event_1.mp4").exists())
            self.assertFalse((old_dir / "event_1.json").exists())
            self.assertTrue((disabled_dir / "cam1" / "20260627" / "event_1.mp4").exists())
            self.assertEqual(list(debug_dir.rglob("*.mp4")), [])
            self.assertEqual(list(debug_dir.rglob("*.json")), [])

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
            self.assertEqual(Path(saved["clip_path"]).parent.parent.name, "cam1")
            self.assertEqual(Path(saved["clip_path"]).stem, "event_1")
            self.assertIsNone(metadata["verification"])

    def test_save_clip_uses_camera_date_directory_and_next_daily_sequence_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ClipBuilder(output_dir=temp_dir)
            existing_dir = Path(temp_dir) / "file_cam_001" / "20260616"
            existing_dir.mkdir(parents=True)
            for index in range(1, 4):
                (existing_dir / f"event_{index}.mp4").write_bytes(b"existing")

            candidate = {
                "camera_id": "file_cam_001",
                "candidate_id": "candidate-with-extra-info",
                "timestamp_ms": 12345,
            }
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frames = [
                {
                    "camera_id": "file_cam_001",
                    "frame_id": 1,
                    "timestamp_ms": 12345,
                    "frame": frame,
                    "width": 16,
                    "height": 16,
                    "fps": 5,
                    "source_uri": "video.mp4",
                }
            ]

            with (
                patch("services.clip_builder.cv2", FakeCv2),
                patch("services.clip_builder.datetime") as fake_datetime,
            ):
                fake_datetime.now.return_value.strftime.return_value = "20260616"
                fake_datetime.now.return_value.isoformat.return_value = "2026-06-16T09:30:00"
                saved = builder.save_event_clip(
                    candidate=candidate,
                    verification={"result": "confirmed_fall"},
                    frame_packets=frames,
                    category="confirmed_fall",
                )

            clip_path = Path(saved["clip_path"])
            metadata_path = Path(saved["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(clip_path, existing_dir / "event_4.mp4")
        self.assertEqual(metadata_path, existing_dir / "event_4.json")
        self.assertEqual(saved["event_id"], "file_cam_001_20260616_event_4")
        self.assertEqual(metadata["category"], "confirmed_fall")
        self.assertEqual(metadata["candidate"]["candidate_id"], "candidate-with-extra-info")

    def test_transcode_browser_mp4_replaces_clip_with_h264_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ClipBuilder(output_dir=temp_dir)
            clip_path = Path(temp_dir) / "event_1.mp4"
            clip_path.write_bytes(b"fmp4")

            def fake_run(command, check, stdout, stderr):
                self.assertTrue(check)
                self.assertIn("libx264", command)
                self.assertIn("yuv420p", command)
                Path(command[-1]).write_bytes(b"h264")

            with patch("services.clip_builder.subprocess.run", side_effect=fake_run):
                builder._transcode_browser_mp4(clip_path)

            self.assertEqual(clip_path.read_bytes(), b"h264")
            self.assertFalse(Path(str(clip_path) + ".tmp.mp4").exists())


if __name__ == "__main__":
    unittest.main()
