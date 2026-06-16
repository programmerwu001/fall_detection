import csv
import json
import tempfile
import unittest
from pathlib import Path

from services.event_repository import EventRepository, JOB_DONE, JOB_PENDING
from services.metrics_report import (
    build_metrics_report,
    render_markdown_report,
    write_metrics_report,
)


class MetricsReportTest(unittest.TestCase):
    def test_build_metrics_report_summarizes_event_metadata_and_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            db_path = root / "records.db"
            _write_event_metadata(
                event_dir / "candidates" / "cam1" / "20260609" / "event1.json",
                event_id="event1",
                category="candidates",
                source_uri="source_a.mp4",
                candidate_timestamp_ms=1000,
                yolo_score=0.82,
                duration_ms=6000,
                frame_count=90,
                verification=None,
            )
            _write_event_metadata(
                event_dir / "confirmed_fall" / "cam1" / "20260609" / "event2.json",
                event_id="event2",
                category="confirmed_fall",
                source_uri="source_b.mp4",
                candidate_timestamp_ms=3000,
                yolo_score=0.91,
                duration_ms=4000,
                frame_count=60,
                verification={"result": "confirmed_fall", "confidence": 0.93},
            )
            repository = EventRepository(db_path).initialize()
            repository.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="source_a.mp4",
                clip_path="event1.mp4",
                metadata_path="event1.json",
                candidate={"candidate_id": "c1"},
                yolo_score=0.82,
            )
            repository.enqueue_vlm_job(event_id="event1", job_id="job1")
            repository.create_candidate_event(
                event_id="event2",
                camera_id="cam1",
                source_uri="source_b.mp4",
                clip_path="event2.mp4",
                metadata_path="event2.json",
                candidate={"candidate_id": "c2"},
                yolo_score=0.91,
            )
            repository.enqueue_vlm_job(event_id="event2", job_id="job2")
            repository.complete_vlm_job(
                job_id="job2",
                verification={"result": "confirmed_fall", "confidence": 0.93},
                final_status="confirmed_fall",
            )

            report = build_metrics_report(
                event_dir=event_dir,
                queue_db_path=db_path,
                generated_at="2026-06-09T10:00:00",
            )

        self.assertEqual(report["events"]["total"], 2)
        self.assertEqual(report["events"]["by_category"]["candidates"], 1)
        self.assertEqual(report["events"]["by_category"]["confirmed_fall"], 1)
        self.assertEqual(report["clips"]["duration_seconds"]["total"], 10.0)
        self.assertEqual(report["clips"]["frames"]["total"], 150)
        self.assertAlmostEqual(report["yolo"]["average_score"], 0.865)
        self.assertEqual(report["vlm"]["verified_events"], 1)
        self.assertEqual(report["vlm"]["confirmed_fall"], 1)
        self.assertEqual(report["queue"]["jobs"][JOB_PENDING], 1)
        self.assertEqual(report["queue"]["jobs"][JOB_DONE], 1)

    def test_label_evaluation_computes_precision_recall_and_time_accuracy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            labels_path = root / "labels.csv"
            _write_event_metadata(
                event_dir / "confirmed_fall" / "cam1" / "20260609" / "event1.json",
                event_id="event1",
                category="confirmed_fall",
                source_uri="source_a.mp4",
                candidate_timestamp_ms=1500,
                yolo_score=0.82,
                duration_ms=6000,
                frame_count=90,
                verification={"result": "confirmed_fall", "confidence": 0.9},
            )
            _write_event_metadata(
                event_dir / "confirmed_fall" / "cam1" / "20260609" / "event2.json",
                event_id="event2",
                category="confirmed_fall",
                source_uri="source_b.mp4",
                candidate_timestamp_ms=9000,
                yolo_score=0.7,
                duration_ms=6000,
                frame_count=90,
                verification={"result": "confirmed_fall", "confidence": 0.8},
            )
            with labels_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["source_uri", "event_start_ms", "event_end_ms"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_uri": "source_a.mp4",
                        "event_start_ms": "1000",
                        "event_end_ms": "5000",
                    }
                )
                writer.writerow(
                    {
                        "source_uri": "source_c.mp4",
                        "event_start_ms": "2000",
                        "event_end_ms": "6000",
                    }
                )

            report = build_metrics_report(
                event_dir=event_dir,
                labels_path=labels_path,
                generated_at="2026-06-09T10:00:00",
            )

        evaluation = report["label_evaluation"]
        self.assertTrue(evaluation["available"])
        self.assertEqual(evaluation["labels"], 2)
        self.assertEqual(evaluation["detections"], 2)
        self.assertEqual(evaluation["true_positives"], 1)
        self.assertEqual(evaluation["false_positives"], 1)
        self.assertEqual(evaluation["false_negatives"], 1)
        self.assertEqual(evaluation["precision"], 0.5)
        self.assertEqual(evaluation["recall"], 0.5)
        self.assertEqual(evaluation["f1"], 0.5)
        self.assertEqual(evaluation["start_time_accuracy"]["within_1000ms"], 0.5)
        self.assertEqual(evaluation["start_time_accuracy"]["within_2000ms"], 0.5)
        self.assertEqual(evaluation["start_time_error_ms"]["mean_abs"], 500.0)

    def test_video_label_evaluation_computes_accuracy_with_negative_videos(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            video_labels_path = root / "video_labels.csv"
            _write_event_metadata(
                event_dir / "confirmed_fall" / "cam1" / "20260609" / "event1.json",
                event_id="event1",
                category="confirmed_fall",
                source_uri=r"C:\videos\FallForwardS1.avi",
                candidate_timestamp_ms=1500,
                yolo_score=0.82,
                duration_ms=6000,
                frame_count=90,
                verification={"result": "confirmed_fall", "confidence": 0.9},
            )
            _write_event_metadata(
                event_dir / "confirmed_fall" / "cam1" / "20260609" / "event2.json",
                event_id="event2",
                category="confirmed_fall",
                source_uri=r"C:\videos\WalkS1.avi",
                candidate_timestamp_ms=9000,
                yolo_score=0.7,
                duration_ms=6000,
                frame_count=90,
                verification={"result": "confirmed_fall", "confidence": 0.8},
            )
            with video_labels_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["source_uri", "has_fall"])
                writer.writeheader()
                writer.writerow({"source_uri": "FallForwardS1.avi", "has_fall": "1"})
                writer.writerow({"source_uri": "FallLeftS1.avi", "has_fall": "1"})
                writer.writerow({"source_uri": "WalkS1.avi", "has_fall": "0"})
                writer.writerow({"source_uri": "HopS1.avi", "has_fall": "0"})

            report = build_metrics_report(
                event_dir=event_dir,
                video_labels_path=video_labels_path,
                generated_at="2026-06-09T10:00:00",
            )

        evaluation = report["video_label_evaluation"]
        self.assertTrue(evaluation["available"])
        self.assertEqual(evaluation["labels"], 4)
        self.assertEqual(evaluation["positive_labels"], 2)
        self.assertEqual(evaluation["negative_labels"], 2)
        self.assertEqual(evaluation["true_positives"], 1)
        self.assertEqual(evaluation["false_positives"], 1)
        self.assertEqual(evaluation["true_negatives"], 1)
        self.assertEqual(evaluation["false_negatives"], 1)
        self.assertEqual(evaluation["accuracy"], 0.5)
        self.assertEqual(evaluation["precision"], 0.5)
        self.assertEqual(evaluation["recall"], 0.5)
        self.assertEqual(evaluation["f1"], 0.5)

    def test_video_label_evaluation_uses_final_vlm_status_for_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            video_labels_path = root / "video_labels.csv"
            _write_event_metadata(
                event_dir / "need_human_review" / "cam1" / "20260609" / "event1.json",
                event_id="event1",
                category="need_human_review",
                source_uri="FallForwardS1.avi",
                candidate_timestamp_ms=1500,
                yolo_score=0.82,
                duration_ms=6000,
                frame_count=90,
                verification={"result": "need_human_review", "confidence": 0.55},
            )
            _write_event_metadata(
                event_dir / "rejected" / "cam1" / "20260609" / "event2.json",
                event_id="event2",
                category="rejected",
                source_uri="WalkS1.avi",
                candidate_timestamp_ms=3000,
                yolo_score=0.72,
                duration_ms=6000,
                frame_count=90,
                verification={"result": "rejected", "confidence": 0.85},
            )
            _write_event_metadata(
                event_dir / "candidates" / "cam1" / "20260609" / "event3.json",
                event_id="event3",
                category="candidates",
                source_uri="FallLeftS1.avi",
                candidate_timestamp_ms=5000,
                yolo_score=0.9,
                duration_ms=6000,
                frame_count=90,
                verification=None,
            )
            with video_labels_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["source_uri", "has_fall"])
                writer.writeheader()
                writer.writerow({"source_uri": "FallForwardS1.avi", "has_fall": "1"})
                writer.writerow({"source_uri": "FallLeftS1.avi", "has_fall": "1"})
                writer.writerow({"source_uri": "WalkS1.avi", "has_fall": "0"})

            report = build_metrics_report(
                event_dir=event_dir,
                video_labels_path=video_labels_path,
                generated_at="2026-06-09T10:00:00",
            )

        evaluation = report["video_label_evaluation"]
        self.assertEqual(evaluation["true_positives"], 1)
        self.assertEqual(evaluation["false_positives"], 0)
        self.assertEqual(evaluation["true_negatives"], 1)
        self.assertEqual(evaluation["false_negatives"], 1)
        self.assertEqual(evaluation["review_positive_predictions"], 1)
        self.assertEqual(evaluation["confirmed_positive_predictions"], 0)
        self.assertEqual(evaluation["rejected_negative_predictions"], 1)
        self.assertEqual(evaluation["pending_sources"], 1)

    def test_write_metrics_report_creates_json_and_markdown_without_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            output_json = root / "metrics_summary.json"
            output_md = root / "metrics_summary.md"
            _write_event_metadata(
                event_dir / "candidates" / "cam1" / "20260609" / "event1.json",
                event_id="event1",
                category="candidates",
                source_uri="source_a.mp4",
                candidate_timestamp_ms=1000,
                yolo_score=0.82,
                duration_ms=6000,
                frame_count=90,
                verification=None,
            )
            report = build_metrics_report(
                event_dir=event_dir,
                generated_at="2026-06-09T10:00:00",
            )

            paths = write_metrics_report(
                report=report,
                output_json=output_json,
                output_markdown=output_md,
            )

            loaded = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(paths["json"], str(output_json))
        self.assertEqual(paths["markdown"], str(output_md))
        self.assertEqual(loaded["events"]["total"], 1)
        self.assertIn("Fall Edge Gateway Metrics Report", markdown)
        self.assertIn("labels file was not provided", markdown)
        self.assertIn("1 candidate clip(s) were queued for VLM review", markdown)
        self.assertIn(
            "labels file was not provided",
            render_markdown_report(report),
        )


def _write_event_metadata(
    path,
    event_id,
    category,
    source_uri,
    candidate_timestamp_ms,
    yolo_score,
    duration_ms,
    frame_count,
    verification,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "event_id": event_id,
        "camera_id": "cam1",
        "clip_path": str(path.with_suffix(".mp4")),
        "metadata_path": str(path),
        "frame_count": frame_count,
        "fps": 15.0,
        "width": 640,
        "height": 480,
        "start_timestamp_ms": max(0, candidate_timestamp_ms - 3000),
        "end_timestamp_ms": candidate_timestamp_ms + 3000,
        "duration_ms": duration_ms,
        "category": category,
        "source_uri": source_uri,
        "candidate": {
            "candidate_id": f"candidate_{event_id}",
            "timestamp_ms": candidate_timestamp_ms,
            "score": yolo_score,
        },
        "verification": verification,
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
