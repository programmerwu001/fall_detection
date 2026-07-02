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
            "clip_path": "private_clip.mp4",
            "debug_clip_path": "debug_clip.mp4",
            "metadata_path": "clip.json",
        }


class FakeEventRepository:
    def __init__(self):
        self.created_events = []
        self.queued_jobs = []
        self.completed_jobs = []

    def create_candidate_event(self, **kwargs):
        self.created_events.append(kwargs)
        return dict(kwargs)

    def enqueue_vlm_job(self, **kwargs):
        self.queued_jobs.append(kwargs)
        return {"job_id": f"vlm_{kwargs['event_id']}", **kwargs}

    def complete_vlm_job(self, **kwargs):
        self.completed_jobs.append(kwargs)
        return {"job_id": kwargs["job_id"], "status": "done"}


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


class FakeFileVideoSource:
    packets_by_uri = {}

    def __init__(self, camera_id, source_uri, fps_limit=None, realtime=False, loop=False):
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.packets = [dict(packet) for packet in self.packets_by_uri[source_uri]]
        self.index = 0

    def open(self):
        return self

    def read(self):
        if self.index >= len(self.packets):
            return None
        packet = dict(self.packets[self.index])
        self.index += 1
        packet.setdefault("camera_id", self.camera_id)
        packet.setdefault("source_uri", self.source_uri)
        packet.setdefault("frame", object())
        packet.setdefault("width", 640)
        packet.setdefault("height", 480)
        packet.setdefault("fps", 10.0)
        return packet

    def close(self):
        pass


class RecordingYoloDetector:
    def __init__(self):
        self.packets = []

    def detect(self, packet):
        self.packets.append(dict(packet))
        return []


class BoundaryCandidateDetector:
    def __init__(self, source_name, frame_id):
        self.source_name = source_name
        self.frame_id = frame_id
        self.reset_count = 0

    def detect(self, packet):
        if (
            Path(packet["source_uri"]).name == self.source_name
            and packet["frame_id"] == self.frame_id
        ):
            return [
                {
                    "camera_id": packet["camera_id"],
                    "candidate_id": "boundary_candidate",
                    "timestamp_ms": packet["timestamp_ms"],
                    "frame_id": packet["frame_id"],
                    "score": 1.0,
                    "source_uri": packet["source_uri"],
                }
            ]
        return []

    def reset_state(self):
        self.reset_count += 1


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

    def test_finalize_event_uses_low_risk_demo_mode_when_vlm_is_skipped(self):
        stats = run_gateway.PipelineStats()
        clip_builder = FakeClipBuilder()
        args = argparse.Namespace(
            skip_vlm=True,
            async_vlm=False,
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

        self.assertEqual(stats.vlm_review, 1)
        self.assertEqual(stats.clips_saved, 1)
        self.assertEqual(clip_builder.saved[0]["category"], "need_human_review")

    def test_skip_vlm_async_demo_creates_low_risk_database_alert_without_queue(self):
        stats = run_gateway.PipelineStats()
        clip_builder = FakeClipBuilder()
        repository = FakeEventRepository()
        args = argparse.Namespace(
            skip_vlm=True,
            async_vlm=True,
            vlm_confidence_threshold=0.6,
            save_review=False,
            save_rejected=False,
        )
        active = run_gateway.ActiveEvent(
            candidate={"camera_id": "cam1", "candidate_id": "c1", "timestamp_ms": 1000},
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
            None,
            clip_builder,
            args,
            stats,
            event_repository=repository,
        )

        self.assertEqual(stats.candidate_events_saved, 1)
        self.assertEqual(stats.vlm_review, 1)
        self.assertEqual(repository.queued_jobs, [])
        self.assertEqual(repository.created_events[0]["clip_path"], "private_clip.mp4")
        self.assertEqual(repository.completed_jobs[0]["verification"]["result"], "need_human_review")
        self.assertEqual(repository.completed_jobs[0]["final_status"], "need_human_review")

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
        self.assertEqual(repository.created_events[0]["clip_path"], "private_clip.mp4")
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

    def test_process_video_sequence_treats_folder_as_one_continuous_camera(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            FakeFileVideoSource.packets_by_uri = {
                str(first): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
                str(second): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
            }
            args = argparse.Namespace(
                fps_limit=10.0,
                realtime=False,
                video_boundary_policy="continuous",
                pre_event_seconds=3.0,
                post_event_seconds=3.0,
                cooldown_seconds=8.0,
                skip_vlm=True,
                async_vlm=False,
                vlm_confidence_threshold=0.6,
                save_review=False,
                save_rejected=False,
            )
            detector = RecordingYoloDetector()
            stats = run_gateway.PipelineStats()

            with patch("run_gateway.FileVideoSource", FakeFileVideoSource):
                run_gateway.process_video_sequence(
                    video_paths=[first, second],
                    camera_id="file_cam_001",
                    args=args,
                    yolo_detector=detector,
                    vlm_verifier=None,
                    event_buffer=EventBuffer(max_seconds=10),
                    clip_builder=FakeClipBuilder(),
                    stats=stats,
                )

        self.assertEqual([packet["camera_id"] for packet in detector.packets], ["file_cam_001"] * 4)
        self.assertEqual([packet["frame_id"] for packet in detector.packets], [0, 1, 2, 3])
        self.assertEqual([packet["timestamp_ms"] for packet in detector.packets], [0, 100, 200, 300])
        self.assertEqual([Path(packet["source_uri"]).name for packet in detector.packets], ["a.mp4", "a.mp4", "b.mp4", "b.mp4"])
        self.assertEqual(stats.frames_read, 4)
        self.assertEqual(stats.videos_processed, 2)

    def test_soft_reset_boundary_resets_frame_id_and_timestamp_for_unrelated_videos(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            FakeFileVideoSource.packets_by_uri = {
                str(first): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
                str(second): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
            }
            args = argparse.Namespace(
                fps_limit=10.0,
                realtime=False,
                video_boundary_policy="soft_reset",
                pre_event_seconds=3.0,
                post_event_seconds=3.0,
                cooldown_seconds=8.0,
                skip_vlm=True,
                async_vlm=False,
                vlm_confidence_threshold=0.6,
                save_review=False,
                save_rejected=False,
            )
            detector = RecordingYoloDetector()

            with patch("run_gateway.FileVideoSource", FakeFileVideoSource):
                run_gateway.process_video_sequence(
                    video_paths=[first, second],
                    camera_id="file_cam_001",
                    args=args,
                    yolo_detector=detector,
                    vlm_verifier=None,
                    event_buffer=EventBuffer(max_seconds=10),
                    clip_builder=FakeClipBuilder(),
                    stats=run_gateway.PipelineStats(),
                )

        self.assertEqual([packet["camera_id"] for packet in detector.packets], ["file_cam_001"] * 4)
        self.assertEqual([packet["frame_id"] for packet in detector.packets], [0, 1, 0, 1])
        self.assertEqual([packet["timestamp_ms"] for packet in detector.packets], [0, 100, 0, 100])
        self.assertEqual(
            [Path(packet["source_uri"]).name for packet in detector.packets],
            ["a.mp4", "a.mp4", "b.mp4", "b.mp4"],
        )

    def test_soft_reset_boundary_does_not_include_previous_video_in_pre_event_clip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            FakeFileVideoSource.packets_by_uri = {
                str(first): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
                str(second): [
                    {"frame_id": 0, "timestamp_ms": 0},
                ],
            }
            args = argparse.Namespace(
                fps_limit=10.0,
                realtime=False,
                video_boundary_policy="soft_reset",
                pre_event_seconds=3.0,
                post_event_seconds=0.0,
                cooldown_seconds=8.0,
                skip_vlm=True,
                async_vlm=False,
                vlm_confidence_threshold=0.6,
                save_review=False,
                save_rejected=False,
            )
            detector = BoundaryCandidateDetector(source_name="b.mp4", frame_id=0)
            clip_builder = FakeClipBuilder()

            with patch("run_gateway.FileVideoSource", FakeFileVideoSource):
                run_gateway.process_video_sequence(
                    video_paths=[first, second],
                    camera_id="file_cam_001",
                    args=args,
                    yolo_detector=detector,
                    vlm_verifier=None,
                    event_buffer=EventBuffer(max_seconds=10),
                    clip_builder=clip_builder,
                    stats=run_gateway.PipelineStats(),
                )

        saved_frames = clip_builder.saved[0]["frame_packets"]
        self.assertEqual([Path(packet["source_uri"]).name for packet in saved_frames], ["b.mp4"])
        self.assertEqual([packet["frame_id"] for packet in saved_frames], [0])
        self.assertEqual([packet["timestamp_ms"] for packet in saved_frames], [0])
        self.assertEqual(detector.reset_count, 1)

    def test_continuous_boundary_keeps_previous_video_in_pre_event_clip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            FakeFileVideoSource.packets_by_uri = {
                str(first): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
                str(second): [
                    {"frame_id": 0, "timestamp_ms": 0},
                ],
            }
            args = argparse.Namespace(
                fps_limit=10.0,
                realtime=False,
                video_boundary_policy="continuous",
                pre_event_seconds=3.0,
                post_event_seconds=0.0,
                cooldown_seconds=8.0,
                skip_vlm=True,
                async_vlm=False,
                vlm_confidence_threshold=0.6,
                save_review=False,
                save_rejected=False,
            )
            detector = BoundaryCandidateDetector(source_name="b.mp4", frame_id=2)
            clip_builder = FakeClipBuilder()

            with patch("run_gateway.FileVideoSource", FakeFileVideoSource):
                run_gateway.process_video_sequence(
                    video_paths=[first, second],
                    camera_id="file_cam_001",
                    args=args,
                    yolo_detector=detector,
                    vlm_verifier=None,
                    event_buffer=EventBuffer(max_seconds=10),
                    clip_builder=clip_builder,
                    stats=run_gateway.PipelineStats(),
                )

        saved_frames = clip_builder.saved[0]["frame_packets"]
        self.assertEqual(
            [Path(packet["source_uri"]).name for packet in saved_frames],
            ["a.mp4", "a.mp4", "b.mp4"],
        )
        self.assertEqual([packet["frame_id"] for packet in saved_frames], [0, 1, 2])
        self.assertEqual(detector.reset_count, 0)

    def test_soft_reset_boundary_finalizes_partial_event_before_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            FakeFileVideoSource.packets_by_uri = {
                str(first): [
                    {"frame_id": 0, "timestamp_ms": 0},
                    {"frame_id": 1, "timestamp_ms": 100},
                ],
                str(second): [
                    {"frame_id": 0, "timestamp_ms": 0},
                ],
            }
            args = argparse.Namespace(
                fps_limit=10.0,
                realtime=False,
                video_boundary_policy="soft_reset",
                pre_event_seconds=3.0,
                post_event_seconds=5.0,
                cooldown_seconds=8.0,
                skip_vlm=True,
                async_vlm=False,
                vlm_confidence_threshold=0.6,
                save_review=False,
                save_rejected=False,
            )
            detector = BoundaryCandidateDetector(source_name="a.mp4", frame_id=1)
            clip_builder = FakeClipBuilder()

            with patch("run_gateway.FileVideoSource", FakeFileVideoSource):
                run_gateway.process_video_sequence(
                    video_paths=[first, second],
                    camera_id="file_cam_001",
                    args=args,
                    yolo_detector=detector,
                    vlm_verifier=None,
                    event_buffer=EventBuffer(max_seconds=10),
                    clip_builder=clip_builder,
                    stats=run_gateway.PipelineStats(),
                )

        saved_frames = clip_builder.saved[0]["frame_packets"]
        self.assertEqual([Path(packet["source_uri"]).name for packet in saved_frames], ["a.mp4", "a.mp4"])
        self.assertEqual([packet["frame_id"] for packet in saved_frames], [0, 1])

    def test_main_processes_selected_folder_as_one_simulated_camera(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            args = argparse.Namespace(
                video_dir=str(root),
                recursive=False,
                max_videos=None,
                camera_prefix="file_cam",
                video_boundary_policy="soft_reset",
                log_level="INFO",
                yolo_model="fake.pt",
                yolo_device=None,
                yolo_imgsz=640,
                yolo_conf=0.25,
                candidate_threshold=0.55,
                cooldown_seconds=8.0,
                skip_vlm=True,
                async_vlm=False,
                buffer_seconds=10.0,
                pre_event_seconds=3.0,
                post_event_seconds=3.0,
                output_dir=str(root / "events"),
            )
            sequence_calls = []

            def record_sequence(**kwargs):
                sequence_calls.append(kwargs)

            with (
                patch("run_gateway.parse_args", return_value=args),
                patch("run_gateway.YoloCandidateDetector", return_value=RecordingYoloDetector()),
                patch("run_gateway.ClipBuilder", return_value=FakeClipBuilder()),
                patch("run_gateway.process_video_sequence", side_effect=record_sequence),
                patch("run_gateway.process_video_file"),
            ):
                exit_code = run_gateway.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(sequence_calls), 1)
        self.assertEqual(sequence_calls[0]["video_paths"], [first, second])
        self.assertEqual(sequence_calls[0]["camera_id"], "file_cam_001")

    def test_parse_args_uses_config_defaults_and_allows_cli_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "detection_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "_注释": "中文说明字段应被忽略，不能影响参数解析。",
                        "video_dir": "from_config",
                        "max_videos": 5,
                        "video_boundary_policy": "continuous",
                        "recursive": True,
                        "fps_limit": 8.0,
                        "candidate_threshold": 0.4,
                        "skip_vlm": True,
                        "vlm_backend": "minicpm_chat",
                        "save_debug_raw_event_copy": False,
                        "high_risk_repeat_seconds": 22,
                        "low_risk_repeat_seconds": 66,
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
        self.assertEqual(args.video_boundary_policy, "continuous")
        self.assertTrue(args.recursive)
        self.assertEqual(args.fps_limit, 8.0)
        self.assertEqual(args.candidate_threshold, 0.7)
        self.assertFalse(args.skip_vlm)
        self.assertEqual(args.vlm_backend, "minicpm_chat")
        self.assertFalse(args.save_debug_raw_event_copy)
        self.assertEqual(args.high_risk_repeat_seconds, 22)
        self.assertEqual(args.low_risk_repeat_seconds, 66)


if __name__ == "__main__":
    unittest.main()
