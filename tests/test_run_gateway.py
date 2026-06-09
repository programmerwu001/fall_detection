import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_gateway
from services.event_buffer import EventBuffer


class FakeYoloDetector:
    def __init__(self, candidate):
        self.candidate = candidate

    def detect(self, packet):
        if packet["frame_id"] == self.candidate["frame_id"]:
            return [dict(self.candidate)]
        return []


class FakeClipBuilder:
    def __init__(self):
        self.saved = []

    def save_event_clip(self, **kwargs):
        self.saved.append(kwargs)
        return {
            "event_id": "event1",
            "camera_id": kwargs["candidate"].get("camera_id", "cam1"),
            "source_uri": kwargs["frame_packets"][0].get("source_uri", ""),
            "clip_path": "clip.mp4",
            "metadata_path": "clip.json",
        }


class FakeEventRepository:
    def __init__(self):
        self.created_events = []
        self.queued_jobs = []

    def create_candidate_event(self, **kwargs):
        self.created_events.append(kwargs)
        return dict(kwargs)

    def enqueue_vlm_job(self, **kwargs):
        self.queued_jobs.append(kwargs)
        return {"job_id": f"vlm_{kwargs['event_id']}", **kwargs}


class FakeVlmVerifier:
    model_id = "fake-vlm"
    backend = "fake"

    def __init__(self):
        self.called = False

    def verify(self, **kwargs):
        self.called = True
        return {
            "result": "confirmed_fall",
            "confidence": 1.0,
            "reason": "sync verifier should not run in async mode",
        }


class RunGatewayTest(unittest.TestCase):
    def test_should_save_event_policy(self):
        self.assertTrue(
            run_gateway.should_save_event(
                "confirmed_fall", 0.7, min_confidence=0.6, save_review=False, save_rejected=False
            )
        )
        self.assertFalse(
            run_gateway.should_save_event(
                "confirmed_fall", 0.5, min_confidence=0.6, save_review=True, save_rejected=True
            )
        )
        self.assertTrue(
            run_gateway.should_save_event(
                "need_human_review", 0.0, min_confidence=0.6, save_review=True, save_rejected=False
            )
        )
        self.assertFalse(
            run_gateway.should_save_event(
                "rejected", 1.0, min_confidence=0.6, save_review=True, save_rejected=False
            )
        )

    def test_start_active_event_uses_pre_event_buffer_and_skips_duplicate_append(self):
        buffer = EventBuffer(max_seconds=5)
        for frame_id, timestamp_ms in enumerate([0, 500, 1000, 1500]):
            buffer.append(
                {
                    "camera_id": "cam1",
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "frame": object(),
                }
            )

        active = run_gateway.start_active_event(
            candidate={"camera_id": "cam1", "candidate_id": "c1", "timestamp_ms": 1000},
            event_buffer=buffer,
            camera_id="cam1",
            pre_event_seconds=0.6,
            post_event_seconds=2.0,
        )
        active.append({"camera_id": "cam1", "frame_id": 2, "timestamp_ms": 1000})

        self.assertEqual([packet["frame_id"] for packet in active.frames], [1, 2])
        self.assertEqual(active.end_timestamp_ms, 3000)

    def test_finalize_event_accepts_yolo_candidate_when_vlm_is_skipped(self):
        stats = run_gateway.PipelineStats()
        clip_builder = FakeClipBuilder()
        args = argparse.Namespace(
            skip_vlm=True,
            vlm_confidence_threshold=0.6,
            save_review=False,
            save_rejected=False,
        )
        active = run_gateway.ActiveEvent(
            candidate={"camera_id": "cam1", "candidate_id": "c1", "timestamp_ms": 1000},
            frames=[
                {"camera_id": "cam1", "frame_id": 0, "timestamp_ms": 1000, "frame": object()}
            ],
            end_timestamp_ms=1000,
        )

        run_gateway.finalize_event(active, None, clip_builder, args, stats)

        self.assertEqual(stats.vlm_confirmed, 1)
        self.assertEqual(stats.clips_saved, 1)
        self.assertEqual(clip_builder.saved[0]["category"], "confirmed_fall")

    def test_finalize_event_in_async_mode_saves_candidate_and_queues_vlm_job(self):
        stats = run_gateway.PipelineStats()
        clip_builder = FakeClipBuilder()
        repository = FakeEventRepository()
        vlm_verifier = FakeVlmVerifier()
        args = argparse.Namespace(
            skip_vlm=False,
            async_vlm=True,
            vlm_confidence_threshold=0.6,
            save_review=False,
            save_rejected=False,
        )
        active = run_gateway.ActiveEvent(
            candidate={
                "camera_id": "cam1",
                "candidate_id": "c1",
                "timestamp_ms": 1000,
                "score": 0.82,
                "source_uri": "video.mp4",
            },
            frames=[
                {
                    "camera_id": "cam1",
                    "frame_id": 0,
                    "timestamp_ms": 1000,
                    "frame": object(),
                    "source_uri": "video.mp4",
                }
            ],
            end_timestamp_ms=1000,
        )

        run_gateway.finalize_event(
            active,
            vlm_verifier,
            clip_builder,
            args,
            stats,
            event_repository=repository,
        )

        self.assertFalse(vlm_verifier.called)
        self.assertEqual(stats.clips_saved, 1)
        self.assertEqual(stats.candidate_events_saved, 1)
        self.assertEqual(stats.vlm_jobs_queued, 1)
        self.assertEqual(clip_builder.saved[0]["verification"], None)
        self.assertEqual(clip_builder.saved[0]["category"], "candidates")
        self.assertEqual(repository.created_events[0]["event_id"], "event1")
        self.assertEqual(repository.created_events[0]["camera_id"], "cam1")
        self.assertEqual(repository.created_events[0]["source_uri"], "video.mp4")
        self.assertEqual(repository.created_events[0]["yolo_score"], 0.82)
        self.assertEqual(repository.queued_jobs[0]["event_id"], "event1")

    def test_scan_video_files_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.txt").write_text("", encoding="utf-8")
            (root / "a.mp4").write_text("", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "c.avi").write_text("", encoding="utf-8")

            non_recursive = run_gateway.scan_video_files(root)
            recursive = run_gateway.scan_video_files(root, recursive=True)

        self.assertEqual([path.name for path in non_recursive], ["a.mp4"])
        self.assertEqual([path.name for path in recursive], ["a.mp4", "c.avi"])

    def test_parse_args_uses_config_defaults_and_allows_cli_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "detection_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "_注释": "中文说明字段应被忽略，不能影响参数解析。",
                        "video_dir": "from_config",
                        "max_videos": 5,
                        "recursive": True,
                        "fps_limit": 8.0,
                        "candidate_threshold": 0.4,
                        "skip_vlm": True,
                        "vlm_backend": "minicpm_chat",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            argv = [
                "run_gateway.py",
                "--config",
                str(config_path),
                "--max-videos",
                "1",
                "--candidate-threshold",
                "0.7",
                "--no-skip-vlm",
            ]
            with patch("sys.argv", argv):
                args = run_gateway.parse_args()

        self.assertEqual(args.config, str(config_path))
        self.assertEqual(args.video_dir, "from_config")
        self.assertEqual(args.max_videos, 1)
        self.assertTrue(args.recursive)
        self.assertEqual(args.fps_limit, 8.0)
        self.assertEqual(args.candidate_threshold, 0.7)
        self.assertFalse(args.skip_vlm)
        self.assertEqual(args.vlm_backend, "minicpm_chat")


if __name__ == "__main__":
    unittest.main()
