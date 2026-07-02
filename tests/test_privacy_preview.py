import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from services.privacy_preview import (
    PersonRegion,
    PrivacyPreviewError,
    PrivacyPreviewGenerator,
    YoloPersonDetector,
    apply_person_silhouettes,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class FakeDetector:
    def __init__(self, regions):
        self.regions = regions
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.regions)


class SequenceDetector:
    def __init__(self, region_sequence):
        self.region_sequence = list(region_sequence)
        self.calls = 0

    def detect(self, frame):
        if self.calls < len(self.region_sequence):
            regions = self.region_sequence[self.calls]
        elif self.region_sequence:
            regions = self.region_sequence[-1]
        else:
            regions = []
        self.calls += 1
        return list(regions)


class FakeYoloModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.float32),
        )
        return [SimpleNamespace(boxes=boxes, masks=None)]


class FakeCapture:
    def __init__(self, frames, fps=5.0):
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeCv2:
    CAP_PROP_FPS = 5
    COLOR_BGR2GRAY = 6
    TERM_CRITERIA_EPS = 2
    TERM_CRITERIA_COUNT = 1

    def __init__(self, frames, writer, flow_shift=(0.0, 0.0)):
        self.capture = FakeCapture(frames)
        self.writer = writer
        self.flow_shift = np.array(flow_shift, dtype=np.float32)

    def VideoCapture(self, path):
        return self.capture

    def VideoWriter(self, path, fourcc, fps, frame_size):
        return self.writer

    def VideoWriter_fourcc(self, *codec):
        return 0

    def cvtColor(self, frame, code):
        return frame[:, :, 0]

    def goodFeaturesToTrack(self, image, mask, maxCorners, qualityLevel, minDistance):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        step = max(1, len(xs) // max(1, min(maxCorners, len(xs))))
        points = np.column_stack([xs[::step], ys[::step]]).astype(np.float32)
        return points[:maxCorners].reshape(-1, 1, 2)

    def calcOpticalFlowPyrLK(self, prev_gray, gray, prev_points, next_points, **kwargs):
        shifted = prev_points.astype(np.float32) + self.flow_shift.reshape(1, 1, 2)
        status = np.ones((len(prev_points), 1), dtype=np.uint8)
        error = np.zeros((len(prev_points), 1), dtype=np.float32)
        return shifted, status, error


class PrivacyPreviewTest(unittest.TestCase):
    def test_apply_person_silhouettes_masks_all_detected_people_and_keeps_background(self):
        frame = np.full((18, 24, 3), 255, dtype=np.uint8)
        mask = np.zeros((18, 24), dtype=np.uint8)
        mask[10:16, 14:21] = 1
        regions = [
            PersonRegion(box=(2, 3, 8, 12)),
            PersonRegion(mask=mask),
        ]

        output = apply_person_silhouettes(frame, regions, color=(4, 5, 6))

        self.assertTrue(np.all(output[3:12, 2:8] == np.array([4, 5, 6], dtype=np.uint8)))
        self.assertTrue(np.all(output[10:16, 14:21] == np.array([4, 5, 6], dtype=np.uint8)))
        self.assertTrue(np.all(output[0:2, 0:2] == 255))

    def test_apply_person_silhouettes_uses_opaque_adaptive_dilation_without_feathering(self):
        frame = np.full((24, 24, 3), 200, dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[8:16, 8:16] = 1

        output = apply_person_silhouettes(
            frame,
            [PersonRegion(mask=mask)],
            color=(0, 0, 0),
        )

        self.assertTrue(np.all(output[9:15, 9:15] == 0))
        self.assertTrue(np.all(output[7:17, 7:17] == 0))
        self.assertTrue(np.all(output[0:4, 0:4] == 200))
        channel = output[:, :, 0]
        self.assertFalse(np.any((channel > 0) & (channel < 200)))

    def test_apply_person_silhouettes_repairs_internal_holes_and_mask_breaks(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:49] = 1
        mask[40:60, 52:60] = 1
        mask[47:53, 54:58] = 0

        output = apply_person_silhouettes(
            frame,
            [PersonRegion(mask=mask)],
            color=(1, 2, 3),
            edge_feather_pixels=0,
        )

        fill = np.array([1, 2, 3], dtype=np.uint8)
        self.assertTrue(np.all(output[50, 50] == fill))
        self.assertTrue(np.all(output[50, 56] == fill))
        self.assertTrue(np.all(output[38, 50] == fill))
        self.assertTrue(np.all(output[50, 38] == fill))
        self.assertTrue(np.all(output[20, 20] == 200))

    def test_yolo_person_detector_uses_high_recall_segmentation_parameters(self):
        model = FakeYoloModel()
        detector = YoloPersonDetector()
        detector._model = model
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        regions = detector.detect(frame)

        self.assertEqual(regions, [])
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertEqual(call["classes"], [0])
        self.assertEqual(call["conf"], 0.12)
        self.assertEqual(call["imgsz"], 960)
        self.assertTrue(call["retina_masks"])
        self.assertFalse(call["verbose"])

    def test_transcode_browser_mp4_replaces_preview_with_h264_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_path = Path(temp_dir) / "privacy_preview.mp4"
            clip_path.write_bytes(b"mp4v")
            generator = PrivacyPreviewGenerator(
                preview_root=temp_dir,
                detector=FakeDetector([]),
                ffmpeg_path="custom-ffmpeg",
            )

            def fake_run(command, check, stdout, stderr):
                self.assertEqual(command[0], "custom-ffmpeg")
                self.assertTrue(check)
                self.assertIn("libx264", command)
                self.assertIn("yuv420p", command)
                Path(command[-1]).write_bytes(b"h264")

            with patch("services.privacy_preview.subprocess.run", side_effect=fake_run):
                generator._transcode_browser_mp4(clip_path)

            self.assertEqual(clip_path.read_bytes(), b"h264")
            self.assertFalse(Path(str(clip_path) + ".h264.tmp.mp4").exists())

    def test_default_ffmpeg_path_uses_current_python_env_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_root = root / "env"
            ffmpeg_path = env_root / "Library" / "bin" / "ffmpeg.exe"
            ffmpeg_path.parent.mkdir(parents=True)
            ffmpeg_path.write_bytes(b"")
            clip_path = root / "privacy_preview.mp4"
            clip_path.write_bytes(b"mp4v")

            with patch("shutil.which", return_value=None), patch("sys.prefix", str(env_root)):
                generator = PrivacyPreviewGenerator(
                    preview_root=temp_dir,
                    detector=FakeDetector([]),
                )

            def fake_run(command, check, stdout, stderr):
                self.assertEqual(Path(command[0]), ffmpeg_path)
                Path(command[-1]).write_bytes(b"h264")

            with patch("services.privacy_preview.subprocess.run", side_effect=fake_run):
                generator._transcode_browser_mp4(clip_path)

            self.assertEqual(clip_path.read_bytes(), b"h264")

    def test_transcode_browser_mp4_requires_ffmpeg_for_browser_playback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_path = Path(temp_dir) / "privacy_preview.mp4"
            clip_path.write_bytes(b"mp4v")
            generator = PrivacyPreviewGenerator(
                preview_root=temp_dir,
                detector=FakeDetector([]),
            )

            with patch(
                "services.privacy_preview.subprocess.run",
                side_effect=FileNotFoundError,
            ):
                with self.assertRaisesRegex(PrivacyPreviewError, "ffmpeg is required"):
                    generator._transcode_browser_mp4(clip_path)

            self.assertEqual(clip_path.read_bytes(), b"mp4v")
            self.assertFalse(Path(str(clip_path) + ".h264.tmp.mp4").exists())

    @unittest.skipIf(cv2 is None, "OpenCV is required for video generation")
    def test_generator_writes_privacy_preview_atomically_under_dedicated_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private_clip = root / "private_events" / "cam1" / "event_1.mp4"
            preview_root = root / "privacy_previews"
            private_clip.parent.mkdir(parents=True)
            _write_test_video(private_clip)
            detector = FakeDetector([PersonRegion(box=(2, 2, 10, 10))])
            generator = PrivacyPreviewGenerator(
                preview_root=preview_root,
                detector=detector,
                codec="mp4v",
            )

            output_path = generator.generate(
                input_path=private_clip,
                event_id="event1",
            )

            self.assertEqual(output_path, preview_root / "event1" / "privacy_preview.mp4")
            self.assertTrue(output_path.exists())
            self.assertFalse(list(output_path.parent.glob("*.tmp.mp4")))
            self.assertGreater(detector.calls, 0)
            self.assertEqual(private_clip.read_bytes()[:4], b"\x00\x00\x00\x1c")

    def test_write_preview_propagates_last_silhouette_when_detector_misses_frame(self):
        frames = [
            np.full((12, 12, 3), 220, dtype=np.uint8),
            np.full((12, 12, 3), 221, dtype=np.uint8),
        ]
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        writer = FakeWriter()
        detector = SequenceDetector(
            [
                [PersonRegion(mask=mask)],
                [],
            ]
        )
        generator = PrivacyPreviewGenerator(detector=detector)
        fake_cv2 = FakeCv2(frames, writer, flow_shift=(5.0, 0.0))

        with patch("services.privacy_preview.cv2", fake_cv2):
            generator._write_preview(Path("input.mp4"), Path("output.mp4"))

        fill = np.array([6, 6, 6], dtype=np.uint8)
        self.assertEqual(len(writer.frames), 2)
        self.assertTrue(np.all(writer.frames[0][2:6, 2:6] == fill))
        self.assertTrue(np.all(writer.frames[1][2:6, 2:6] == 221))
        self.assertTrue(np.all(writer.frames[1][2:6, 7:11] == fill))
        self.assertFalse(np.array_equal(writer.frames[1], frames[1]))

    def test_write_preview_merges_suspiciously_shrunken_mask_with_history(self):
        frames = [
            np.full((24, 24, 3), 220, dtype=np.uint8),
            np.full((24, 24, 3), 221, dtype=np.uint8),
        ]
        large_mask = np.zeros((24, 24), dtype=np.uint8)
        large_mask[4:18, 4:18] = 1
        tiny_mask = np.zeros((24, 24), dtype=np.uint8)
        tiny_mask[10:12, 10:12] = 1
        writer = FakeWriter()
        detector = SequenceDetector(
            [
                [PersonRegion(mask=large_mask)],
                [PersonRegion(mask=tiny_mask)],
            ]
        )
        generator = PrivacyPreviewGenerator(detector=detector)
        fake_cv2 = FakeCv2(frames, writer)

        with patch("services.privacy_preview.cv2", fake_cv2):
            generator._write_preview(Path("input.mp4"), Path("output.mp4"))

        fill = np.array([6, 6, 6], dtype=np.uint8)
        self.assertEqual(len(writer.frames), 2)
        self.assertTrue(np.all(writer.frames[1][5, 5] == fill))
        self.assertTrue(np.all(writer.frames[1][11, 11] == fill))

    def test_write_preview_masks_full_frame_before_first_detection(self):
        frames = [np.full((10, 10, 3), 220, dtype=np.uint8)]
        writer = FakeWriter()
        generator = PrivacyPreviewGenerator(detector=SequenceDetector([[]]))
        fake_cv2 = FakeCv2(frames, writer)

        with patch("services.privacy_preview.cv2", fake_cv2):
            generator._write_preview(Path("input.mp4"), Path("output.mp4"))

        self.assertEqual(len(writer.frames), 1)
        self.assertTrue(np.all(writer.frames[0] == np.array([6, 6, 6], dtype=np.uint8)))


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (16, 16),
    )
    try:
        frame = np.full((16, 16, 3), 220, dtype=np.uint8)
        writer.write(frame)
        writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    unittest.main()
