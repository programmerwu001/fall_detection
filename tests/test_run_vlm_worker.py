import tempfile
import unittest
from pathlib import Path

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

    def test_process_next_job_marks_failed_job_for_retry_when_vlm_errors(self):
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
        self.assertEqual(event["status"], event_state.VLM_PENDING)
        self.assertEqual(job["status"], JOB_PENDING)
        self.assertEqual(job["attempts"], 1)
        self.assertIn("vlm crashed", job["last_error"])


def _create_repository(temp_dir):
    db_path = Path(temp_dir) / "events.db"
    return EventRepository(db_path).initialize()


if __name__ == "__main__":
    unittest.main()
