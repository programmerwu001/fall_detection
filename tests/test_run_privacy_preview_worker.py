import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_privacy_preview_worker
from services import event_state
from services.event_repository import EventRepository, JOB_DONE, JOB_FAILED


class FakePreviewGenerator:
    def __init__(self, preview_root, error=None):
        self.preview_root = Path(preview_root)
        self.error = error
        self.calls = []

    def generate(self, input_path, event_id):
        self.calls.append({"input_path": str(input_path), "event_id": event_id})
        if self.error is not None:
            raise self.error
        output = self.preview_root / event_id / "privacy_preview.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"preview")
        return output


class FakeRepository:
    def initialize(self):
        return self


class FakeDetector:
    def load(self):
        return self


class RunPrivacyPreviewWorkerTest(unittest.TestCase):
    def test_main_logs_resolved_person_model(self):
        args = SimpleNamespace(
            queue_db_path="records.db",
            preview_dir="privacy_previews",
            worker_id="privacy-worker",
            lease_seconds=300,
            poll_interval_seconds=0.0,
            max_retries=1,
            once=True,
            max_jobs=0,
            person_model="models/yolo11m-seg.pt",
            person_confidence=0.31,
            codec="mp4v",
            ffmpeg_path="ffmpeg",
            log_level="INFO",
        )
        repository = FakeRepository()
        detector = FakeDetector()
        generator = object()

        with patch("run_privacy_preview_worker.parse_args", return_value=args):
            with patch("run_privacy_preview_worker.configure_logging"):
                with patch("run_privacy_preview_worker.EventRepository", return_value=repository):
                    with patch("run_privacy_preview_worker.YoloPersonDetector", return_value=detector):
                        with patch(
                            "run_privacy_preview_worker.PrivacyPreviewGenerator",
                            return_value=generator,
                        ):
                            with patch(
                                "run_privacy_preview_worker.run_worker_loop",
                                return_value=run_privacy_preview_worker.WorkerStats(),
                            ):
                                with self.assertLogs(
                                    run_privacy_preview_worker.logger,
                                    level="INFO",
                                ) as captured:
                                    result = run_privacy_preview_worker.main()

        self.assertEqual(result, 0)
        self.assertIn(
            "Privacy preview person model: models/yolo11m-seg.pt",
            "\n".join(captured.output),
        )

    def test_parse_args_accepts_explicit_ffmpeg_path(self):
        with patch(
            "sys.argv",
            [
                "run_privacy_preview_worker.py",
                "--ffmpeg-path",
                r"E:\tools\ffmpeg.exe",
            ],
        ):
            args = run_privacy_preview_worker.parse_args()

        self.assertEqual(args.ffmpeg_path, r"E:\tools\ffmpeg.exe")

    def test_parse_args_uses_config_privacy_preview_model_and_allows_cli_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "detection_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "privacy_preview_model": "models/custom-silhouette-yolo.pt",
                        "person_confidence": 0.41,
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "run_privacy_preview_worker.py",
                    "--config",
                    str(config_path),
                ],
            ):
                args = run_privacy_preview_worker.parse_args()

            self.assertEqual(args.config, str(config_path))
            self.assertEqual(args.person_model, "models/custom-silhouette-yolo.pt")
            self.assertEqual(args.person_confidence, 0.41)

            with patch(
                "sys.argv",
                [
                    "run_privacy_preview_worker.py",
                    "--config",
                    str(config_path),
                    "--person-model",
                    "models/cli-silhouette-yolo.pt",
                ],
            ):
                args = run_privacy_preview_worker.parse_args()

            self.assertEqual(args.person_model, "models/cli-silhouette-yolo.pt")

    def test_process_next_job_reads_private_clip_and_marks_preview_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private_clip = root / "private_events" / "cam1" / "event_1.mp4"
            debug_clip = root / "events" / "cam1" / "event_1.mp4"
            private_clip.parent.mkdir(parents=True)
            debug_clip.parent.mkdir(parents=True)
            private_clip.write_bytes(b"private")
            debug_clip.write_bytes(b"debug")
            repo = _create_alert_repository(root, private_clip)
            generator = FakePreviewGenerator(root / "privacy_previews")

            stats = run_privacy_preview_worker.process_next_job(
                repository=repo,
                generator=generator,
                worker_id="privacy-worker",
                lease_seconds=60,
                max_retries=1,
            )

            event = repo.get_event("event1")
            job = repo.get_privacy_preview_job("privacy_event1")
            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.completed, 1)
            self.assertEqual(generator.calls, [{"input_path": str(private_clip), "event_id": "event1"}])
            self.assertEqual(event["alert_status"], "pending")
            self.assertEqual(event["privacy_preview_status"], "ready")
            self.assertEqual(Path(event["privacy_preview_path"]).parent.parent.name, "privacy_previews")
            self.assertEqual(job["status"], JOB_DONE)

    def test_process_next_job_failure_does_not_change_pending_alert_or_add_preview_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private_clip = root / "private_events" / "cam1" / "event_1.mp4"
            private_clip.parent.mkdir(parents=True)
            private_clip.write_bytes(b"private")
            repo = _create_alert_repository(root, private_clip)
            generator = FakePreviewGenerator(
                root / "privacy_previews",
                error=RuntimeError("mask model failed"),
            )

            with self.assertLogs(run_privacy_preview_worker.logger, level="ERROR"):
                stats = run_privacy_preview_worker.process_next_job(
                    repository=repo,
                    generator=generator,
                    worker_id="privacy-worker",
                    lease_seconds=60,
                    max_retries=1,
                )

            event = repo.get_event("event1")
            job = repo.get_privacy_preview_job("privacy_event1")
            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.failed, 1)
            self.assertEqual(event["alert_status"], "pending")
            self.assertEqual(event["risk_level"], "high_risk")
            self.assertEqual(event["privacy_preview_status"], "failed")
            self.assertIsNone(event["privacy_preview_path"])
            self.assertEqual(job["status"], JOB_FAILED)


def _create_alert_repository(root: Path, private_clip: Path) -> EventRepository:
    repo = EventRepository(root / "records.db").initialize()
    repo.create_candidate_event(
        event_id="event1",
        camera_id="cam1",
        source_uri="source.mp4",
        clip_path=str(private_clip),
        metadata_path=str(private_clip.with_suffix(".json")),
        candidate={"camera_id": "cam1", "candidate_id": "candidate1"},
    )
    repo.enqueue_vlm_job("event1", job_id="vlm_event1")
    repo.complete_vlm_job(
        "vlm_event1",
        verification={"result": event_state.CONFIRMED_FALL, "confidence": 0.9},
        final_status=event_state.CONFIRMED_FALL,
    )
    return repo


if __name__ == "__main__":
    unittest.main()
