import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from services import event_state
from services.event_repository import EventRepository, JOB_PENDING


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
            self.assertEqual(event["risk_level"], "high_risk")
            self.assertEqual(event["alert_status"], "pending")
            self.assertEqual(event["decision_source"], "vlm")
            self.assertFalse(event["system_degraded"])
            self.assertEqual(event["vlm_status"], "confirmed_fall")
            self.assertIsNotNone(event["next_remind_at"])
            preview_job = repo.get_privacy_preview_job("privacy_event1")
            self.assertIsNotNone(preview_job)
            self.assertEqual(preview_job["event_id"], "event1")
            self.assertEqual(preview_job["status"], JOB_PENDING)
            self.assertEqual(event["privacy_preview_status"], "pending")
            self.assertEqual(job["status"], "done")
            self.assertIsNone(job["locked_by"])

    def test_complete_vlm_job_does_not_create_alert_for_rejected_event(self):
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
                    "result": event_state.REJECTED,
                    "confidence": 0.88,
                    "reason": "person is sitting",
                },
                final_status=event_state.REJECTED,
            )

            event = repo.get_event("event1")
            self.assertEqual(event["risk_level"], "no_alarm")
            self.assertEqual(event["alert_status"], "none")
            self.assertIsNone(event["next_remind_at"])
            self.assertIsNone(repo.get_privacy_preview_job("privacy_event1"))
            self.assertEqual(event["privacy_preview_status"], "not_generated")

    def test_due_reminders_skip_handled_events_after_atomic_update(self):
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
                verification={"result": event_state.CONFIRMED_FALL, "confidence": 0.91},
                final_status=event_state.CONFIRMED_FALL,
            )

            first = repo.claim_due_reminders(now="2999-01-01T10:00:00")
            handled = repo.mark_event_handled(
                event_id="event1",
                handled_by="caregiver_a",
                handled_at="2999-01-01T10:00:01",
            )
            second = repo.claim_due_reminders(now="2999-01-01T10:00:30")

            self.assertEqual([event["event_id"] for event in first], ["event1"])
            self.assertEqual(handled["alert_status"], "handled")
            self.assertEqual(handled["handled_by"], "caregiver_a")
            self.assertEqual(second, [])

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
            event = repo.get_event("event1")
            self.assertEqual(event["status"], event_state.NEED_HUMAN_REVIEW)
            self.assertEqual(event["risk_level"], "low_risk")
            self.assertEqual(event["alert_status"], "pending")
            self.assertEqual(event["decision_source"], "yolo_fallback")
            self.assertTrue(event["system_degraded"])
            self.assertEqual(event["vlm_status"], "failed")
            self.assertEqual(event["privacy_preview_status"], "pending")
            self.assertEqual(repo.get_privacy_preview_job("privacy_event1")["status"], JOB_PENDING)

    def test_mark_vlm_job_degraded_creates_low_risk_yolo_fallback_alert(self):
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

            job = repo.mark_vlm_job_degraded(
                job_id=leased["job_id"],
                reason="decision deadline exceeded",
                failure_status="timeout",
            )

            event = repo.get_event("event1")
            self.assertEqual(job["status"], "failed")
            self.assertEqual(event["status"], event_state.NEED_HUMAN_REVIEW)
            self.assertEqual(event["risk_level"], "low_risk")
            self.assertEqual(event["alert_status"], "pending")
            self.assertEqual(event["decision_source"], "yolo_fallback")
            self.assertTrue(event["system_degraded"])
            self.assertEqual(event["vlm_status"], "timeout")
            self.assertIsNotNone(event["next_remind_at"])
            self.assertEqual(event["privacy_preview_status"], "pending")
            self.assertEqual(repo.get_privacy_preview_job("privacy_event1")["status"], JOB_PENDING)

    def test_privacy_preview_job_lifecycle_is_separate_from_alert_status(self):
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
            leased_vlm = repo.lease_vlm_job(worker_id="worker-a", lease_seconds=60)
            repo.complete_vlm_job(
                job_id=leased_vlm["job_id"],
                verification={"result": event_state.CONFIRMED_FALL, "confidence": 0.91},
                final_status=event_state.CONFIRMED_FALL,
            )

            leased_preview = repo.lease_privacy_preview_job(
                worker_id="privacy-worker",
                lease_seconds=60,
            )
            handled = repo.mark_event_handled("event1", "caregiver_a")
            repo.fail_privacy_preview_job(
                job_id=leased_preview["job_id"],
                error="decoder failed",
                max_retries=1,
            )
            event = repo.get_event("event1")

            self.assertEqual(handled["alert_status"], "handled")
            self.assertEqual(event["alert_status"], "handled")
            self.assertEqual(event["handled_by"], "caregiver_a")
            self.assertEqual(event["privacy_preview_status"], "failed")
            self.assertEqual(event["privacy_preview_error"], "decoder failed")
            self.assertIsNone(event["next_remind_at"])

    def test_complete_privacy_preview_job_records_dedicated_preview_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preview_path = Path(temp_dir) / "privacy_previews" / "event1" / "privacy_preview.mp4"
            repo = EventRepository(Path(temp_dir) / "records.db").initialize()
            repo.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="video.mp4",
                clip_path="candidate.mp4",
                metadata_path="candidate.json",
                candidate={"candidate_id": "c1"},
            )
            repo.record_event_decision(
                event_id="event1",
                verification={"result": event_state.NEED_HUMAN_REVIEW, "confidence": 0.44},
                final_status=event_state.NEED_HUMAN_REVIEW,
            )
            leased = repo.lease_privacy_preview_job("privacy-worker", 60)

            repo.complete_privacy_preview_job(
                job_id=leased["job_id"],
                preview_path=str(preview_path),
            )

            event = repo.get_event("event1")
            job = repo.get_privacy_preview_job(leased["job_id"])
            self.assertEqual(event["privacy_preview_status"], "ready")
            self.assertEqual(event["privacy_preview_path"], str(preview_path))
            self.assertEqual(job["status"], "done")

    def test_late_vlm_result_does_not_retrigger_handled_degraded_alert(self):
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
            repo.mark_vlm_job_degraded(
                job_id=leased["job_id"],
                reason="decision deadline exceeded",
                failure_status="timeout",
            )
            repo.mark_event_handled(
                event_id="event1",
                handled_by="caregiver_a",
                handled_at="2999-01-01T10:00:01",
            )

            repo.complete_vlm_job(
                job_id=leased["job_id"],
                verification={"result": event_state.CONFIRMED_FALL, "confidence": 0.99},
                final_status=event_state.CONFIRMED_FALL,
            )

            event = repo.get_event("event1")
            self.assertEqual(event["alert_status"], "handled")
            self.assertEqual(event["handled_by"], "caregiver_a")
            self.assertEqual(event["risk_level"], "low_risk")
            self.assertEqual(event["decision_source"], "yolo_fallback")
            self.assertIsNone(event["next_remind_at"])

    def test_reminder_intervals_come_from_repository_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = EventRepository(
                Path(temp_dir) / "records.db",
                high_risk_repeat_seconds=7,
                low_risk_repeat_seconds=11,
            ).initialize()
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
                verification={"result": event_state.NEED_HUMAN_REVIEW, "confidence": 0.4},
                final_status=event_state.NEED_HUMAN_REVIEW,
            )

            reminders = repo.claim_due_reminders(now="2999-01-01T10:00:00")

            self.assertEqual([event["event_id"] for event in reminders], ["event1"])
            next_remind_at = datetime.fromisoformat(
                repo.get_event("event1")["next_remind_at"]
            )
            self.assertEqual(
                next_remind_at,
                datetime.fromisoformat("2999-01-01T10:00:11"),
            )


if __name__ == "__main__":
    unittest.main()
