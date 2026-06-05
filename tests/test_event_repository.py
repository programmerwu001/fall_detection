import tempfile
import unittest
from pathlib import Path

from services import event_state
from services.event_repository import EventRepository


class EventRepositoryTest(unittest.TestCase):
    def test_create_candidate_event_and_enqueue_vlm_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EventRepository(Path(temp_dir) / "records.db").initialize()

            event = repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="video.mp4",
                clip_path="candidate.mp4",
                metadata_path="candidate.json",
                candidate={"candidate_id": "c1", "score": 0.72},
                yolo_score=0.72,
            )
            job = repo.enqueue_vlm_job(event_id="event1", priority=50)

            self.assertEqual(event["status"], event_state.VLM_PENDING)
            self.assertEqual(event["privacy_status"], "raw_unprotected")
            self.assertEqual(event["candidate"]["candidate_id"], "c1")
            self.assertEqual(job["event_id"], "event1")
            self.assertEqual(job["status"], "pending")
            self.assertEqual(repo.get_queue_stats()["pending"], 1)

    def test_lease_vlm_job_marks_one_job_processing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EventRepository(Path(temp_dir) / "records.db").initialize()
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="video.mp4",
                clip_path="candidate.mp4",
                metadata_path="candidate.json",
                candidate={"candidate_id": "c1"},
            )
            repo.enqueue_vlm_job(event_id="event1")

            leased = repo.lease_vlm_job(worker_id="worker-a", lease_seconds=60)
            second_lease = repo.lease_vlm_job(worker_id="worker-b", lease_seconds=60)

            self.assertIsNotNone(leased)
            self.assertEqual(leased["event_id"], "event1")
            self.assertEqual(leased["status"], "processing")
            self.assertEqual(leased["locked_by"], "worker-a")
            self.assertEqual(leased["attempts"], 1)
            self.assertIsNone(second_lease)
            self.assertEqual(repo.get_event("event1")["status"], event_state.VLM_PROCESSING)

    def test_complete_vlm_job_updates_event_and_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EventRepository(Path(temp_dir) / "records.db").initialize()
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="video.mp4",
                clip_path="candidate.mp4",
                metadata_path="candidate.json",
                candidate={"candidate_id": "c1"},
            )
            repo.enqueue_vlm_job(event_id="event1")
            leased = repo.lease_vlm_job(worker_id="worker-a", lease_seconds=60)

            repo.complete_vlm_job(
                job_id=leased["job_id"],
                verification={
                    "result": event_state.CONFIRMED_FALL,
                    "confidence": 0.91,
                    "reason": "person fell and stayed down",
                },
                final_status=event_state.CONFIRMED_FALL,
            )

            event = repo.get_event("event1")
            job = repo.get_job(leased["job_id"])
            self.assertEqual(event["status"], event_state.CONFIRMED_FALL)
            self.assertEqual(event["verification"]["confidence"], 0.91)
            self.assertEqual(job["status"], "done")
            self.assertIsNone(job["locked_by"])

    def test_fail_vlm_job_retries_then_marks_event_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EventRepository(Path(temp_dir) / "records.db").initialize()
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="video.mp4",
                clip_path="candidate.mp4",
                metadata_path="candidate.json",
                candidate={"candidate_id": "c1"},
            )
            repo.enqueue_vlm_job(event_id="event1")
            leased = repo.lease_vlm_job(worker_id="worker-a", lease_seconds=60)

            retry_job = repo.fail_vlm_job(
                job_id=leased["job_id"],
                error="temporary model error",
                max_retries=2,
            )
            self.assertEqual(retry_job["status"], "pending")
            self.assertEqual(repo.get_event("event1")["status"], event_state.VLM_PENDING)

            leased_again = repo.lease_vlm_job(worker_id="worker-a", lease_seconds=60)
            failed_job = repo.fail_vlm_job(
                job_id=leased_again["job_id"],
                error="model failed again",
                max_retries=2,
            )

            self.assertEqual(failed_job["status"], "failed")
            self.assertEqual(failed_job["last_error"], "model failed again")
            self.assertEqual(repo.get_event("event1")["status"], event_state.NEED_HUMAN_REVIEW)


if __name__ == "__main__":
    unittest.main()
