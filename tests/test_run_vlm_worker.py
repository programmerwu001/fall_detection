import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import run_vlm_worker
from services import event_state
from services.event_repository import EventRepository, JOB_DONE, JOB_PENDING


class FakeVerifier:
    def __init__(self, verification=None, error=None):
        self.verification = verification or {
            "result": event_state.CONFIRMED_FALL,
            "confidence": 0.91,
            "reason": "person fell and remained on the floor",
            "visible_evidence": ["standing to lying transition"],
        }
        self.error = error
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.verification)


class RunVlmWorkerTest(unittest.TestCase):
    def test_default_decision_deadline_is_two_minutes(self):
        defaults = run_vlm_worker._default_arg_values()

        self.assertEqual(defaults["decision_deadline_seconds"], 120.0)

    def test_main_preloads_vlm_before_polling_jobs(self):
        calls = []

        class MainVerifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def load(self):
                calls.append("load")
                return self

        class MainRepository:
            def __init__(self, db_path):
                self.db_path = db_path

            def initialize(self):
                calls.append("repo")
                return self

        def fake_run_worker_loop(**kwargs):
            calls.append("loop")
            return run_vlm_worker.WorkerStats()

        args = Namespace(
            queue_db_path="events.db",
            worker_id="worker1",
            lease_seconds=60,
            poll_interval_seconds=1.0,
            max_retries=2,
            decision_deadline_seconds=45.0,
            once=True,
            max_jobs=None,
            vlm_model="model",
            vlm_backend="transformers",
            vlm_max_frames=12,
            vlm_max_new_tokens=256,
            vlm_temperature=0.0,
            log_level="INFO",
        )

        with patch.object(run_vlm_worker, "parse_args", return_value=args), patch.object(
            run_vlm_worker, "EventRepository", MainRepository
        ), patch.object(
            run_vlm_worker, "VideoVLMVerifier", MainVerifier
        ), patch.object(
            run_vlm_worker, "run_worker_loop", side_effect=fake_run_worker_loop
        ):
            result = run_vlm_worker.main()

        self.assertEqual(result, 0)
        self.assertEqual(calls, ["repo", "load", "loop"])

    def test_process_next_job_calls_vlm_and_writes_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = _create_repository(temp_dir)
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="source.mp4",
                clip_path="clip.mp4",
                metadata_path="clip.json",
                candidate={
                    "camera_id": "cam1",
                    "candidate_id": "candidate1",
                    "timestamp_ms": 1000,
                    "score": 0.82,
                },
                yolo_score=0.82,
            )
            repo.enqueue_vlm_job(event_id="event1", job_id="job1")
            verifier = FakeVerifier()

            stats = run_vlm_worker.process_next_job(
                repository=repo,
                verifier=verifier,
                worker_id="worker1",
                lease_seconds=60,
                max_retries=2,
            )

            event = repo.get_event("event1")
            job = repo.get_job("job1")

        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.completed, 1)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(event["status"], event_state.CONFIRMED_FALL)
        self.assertEqual(event["verification"]["confidence"], 0.91)
        self.assertEqual(job["status"], JOB_DONE)
        self.assertEqual(job["attempts"], 1)
        self.assertEqual(verifier.calls[0]["candidate"]["candidate_id"], "candidate1")
        self.assertEqual(verifier.calls[0]["clip_path"], "clip.mp4")

    def test_process_next_job_marks_low_risk_degraded_alert_when_vlm_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = _create_repository(temp_dir)
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="source.mp4",
                clip_path="clip.mp4",
                metadata_path="clip.json",
                candidate={"camera_id": "cam1", "candidate_id": "candidate1"},
            )
            repo.enqueue_vlm_job(event_id="event1", job_id="job1")
            verifier = FakeVerifier(error=RuntimeError("vlm crashed"))

            with self.assertLogs(run_vlm_worker.logger, level="ERROR") as logs:
                stats = run_vlm_worker.process_next_job(
                    repository=repo,
                    verifier=verifier,
                    worker_id="worker1",
                    lease_seconds=60,
                    max_retries=2,
                )

            event = repo.get_event("event1")
            job = repo.get_job("job1")

        self.assertIn("VLM job failed", logs.output[0])
        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.completed, 0)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(event["status"], event_state.NEED_HUMAN_REVIEW)
        self.assertEqual(event["risk_level"], "low_risk")
        self.assertEqual(event["alert_status"], "pending")
        self.assertEqual(event["decision_source"], "yolo_fallback")
        self.assertTrue(event["system_degraded"])
        self.assertEqual(event["vlm_status"], "failed")
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["attempts"], 1)
        self.assertIn("vlm crashed", job["last_error"])

    def test_process_next_job_degrades_when_decision_deadline_already_expired(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = _create_repository(temp_dir)
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="source.mp4",
                clip_path="private_clip.mp4",
                metadata_path="clip.json",
                candidate={"camera_id": "cam1", "candidate_id": "candidate1"},
            )
            repo.enqueue_vlm_job(event_id="event1", job_id="job1")
            verifier = FakeVerifier()

            with self.assertLogs(run_vlm_worker.logger, level="ERROR") as logs:
                stats = run_vlm_worker.process_next_job(
                    repository=repo,
                    verifier=verifier,
                    worker_id="worker1",
                    lease_seconds=60,
                    max_retries=2,
                    decision_deadline_seconds=0,
                )

            event = repo.get_event("event1")
            job = repo.get_job("job1")

        self.assertIn("VLM job failed", logs.output[0])
        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(verifier.calls, [])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(event["risk_level"], "low_risk")
        self.assertEqual(event["decision_source"], "yolo_fallback")
        self.assertEqual(event["vlm_status"], "timeout")

    def test_process_next_job_gives_aged_pending_job_full_deadline_when_leased(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = _create_repository(temp_dir)
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="source.mp4",
                clip_path="private_clip.mp4",
                metadata_path="clip.json",
                candidate={"camera_id": "cam1", "candidate_id": "candidate1"},
            )
            repo.enqueue_vlm_job(event_id="event1", job_id="job1")
            connection = repo._connect()
            try:
                connection.execute(
                    "UPDATE vlm_jobs SET created_at = ? WHERE job_id = ?",
                    ("2000-01-01T00:00:00", "job1"),
                )
                connection.commit()
            finally:
                connection.close()
            verifier = FakeVerifier()

            stats = run_vlm_worker.process_next_job(
                repository=repo,
                verifier=verifier,
                worker_id="worker1",
                lease_seconds=60,
                max_retries=2,
                decision_deadline_seconds=120.0,
            )

            event = repo.get_event("event1")
            job = repo.get_job("job1")

        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.completed, 1)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(verifier.calls[0]["candidate"]["candidate_id"], "candidate1")
        self.assertEqual(job["status"], JOB_DONE)
        self.assertEqual(event["status"], event_state.CONFIRMED_FALL)

    def test_process_next_job_degrades_when_verifier_call_exceeds_timeout(self):
        class SlowVerifier(FakeVerifier):
            def __init__(self):
                super().__init__()
                self.finished = False

            def verify(self, **kwargs):
                self.calls.append(kwargs)
                time.sleep(1.0)
                self.finished = True
                return dict(self.verification)

        with tempfile.TemporaryDirectory() as temp_dir:
            repo = _create_repository(temp_dir)
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="source.mp4",
                clip_path="private_clip.mp4",
                metadata_path="clip.json",
                candidate={"camera_id": "cam1", "candidate_id": "candidate1"},
            )
            repo.enqueue_vlm_job(event_id="event1", job_id="job1")
            connection = repo._connect()
            try:
                connection.execute(
                    "UPDATE vlm_jobs SET created_at = ? WHERE job_id = ?",
                    ("2999-01-01T00:00:00", "job1"),
                )
                connection.commit()
            finally:
                connection.close()
            verifier = SlowVerifier()

            with self.assertLogs(run_vlm_worker.logger, level="ERROR") as logs:
                stats = run_vlm_worker.process_next_job(
                    repository=repo,
                    verifier=verifier,
                    worker_id="worker1",
                    lease_seconds=60,
                    max_retries=2,
                    decision_deadline_seconds=0.5,
                )

            event = repo.get_event("event1")

        self.assertIn("VLM job failed", logs.output[0])
        self.assertEqual(stats.processed, 1)
        self.assertEqual(stats.failed, 1)
        self.assertTrue(verifier.finished)
        self.assertEqual(event["risk_level"], "low_risk")
        self.assertEqual(event["decision_source"], "yolo_fallback")
        self.assertEqual(event["vlm_status"], "timeout")


def _create_repository(temp_dir):
    db_path = Path(temp_dir) / "events.db"
    return EventRepository(db_path).initialize()


if __name__ == "__main__":
    unittest.main()
