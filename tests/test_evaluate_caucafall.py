import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_caucafall import build_caucafall_report, render_markdown, write_outputs
from services.event_repository import EventRepository


class EvaluateCaucafallTest(unittest.TestCase):
    def test_build_caucafall_report_summarizes_vlm_results_and_video_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            labels_path = root / "video_labels.csv"
            _write_event(
                event_dir / "confirmed_fall" / "cam1" / "20260616" / "event1.json",
                source_uri="FallForwardS1.avi",
                category="confirmed_fall",
                yolo_score=0.9,
                verification={
                    "result": "confirmed_fall",
                    "confidence": 0.88,
                    "reason": "老人从站立转为倒地并停留在地面。",
                },
            )
            _write_event(
                event_dir / "need_human_review" / "cam1" / "20260616" / "event2.json",
                source_uri="WalkS1.avi",
                category="need_human_review",
                yolo_score=0.6,
                verification={
                    "result": "need_human_review",
                    "confidence": 0.45,
                    "reason": "姿态不清晰，需要人工确认。",
                },
            )
            _write_event(
                event_dir / "rejected" / "cam1" / "20260616" / "event3.json",
                source_uri="HopS1.avi",
                category="rejected",
                yolo_score=0.4,
                verification={
                    "result": "rejected",
                    "confidence": 0.91,
                    "reason": "画面中人员保持站立，没有倒地。",
                },
            )
            _write_video_labels(
                labels_path,
                [
                    ("FallForwardS1.avi", 1),
                    ("FallLeftS1.avi", 1),
                    ("WalkS1.avi", 0),
                    ("HopS1.avi", 0),
                ],
            )

            report = build_caucafall_report(event_dir, labels_path)

        self.assertEqual(report["videos"]["total"], 4)
        self.assertEqual(report["videos"]["fall"], 2)
        self.assertEqual(report["videos"]["nofall"], 2)
        self.assertEqual(report["vlm"]["confirmed_fall"], 1)
        self.assertEqual(report["vlm"]["need_human_review"], 1)
        self.assertEqual(report["vlm"]["rejected"], 1)
        self.assertEqual(report["yolo"]["candidates"], 3)
        self.assertEqual(report["yolo"]["average_score"], 0.633)
        metrics = report["metrics"]
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        details = {row["source_uri"]: row for row in report["details"]}
        self.assertEqual(details["WalkS1.avi"]["outcome"], "FP")
        self.assertEqual(details["WalkS1.avi"]["vlm_reason"], "姿态不清晰，需要人工确认。")

    def test_build_caucafall_report_uses_sqlite_vlm_results_after_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            labels_path = root / "video_labels.csv"
            db_path = root / "records.db"
            _write_event(
                event_dir / "cam1" / "20260616" / "event1.json",
                source_uri="FallForwardS1.avi",
                category="candidates",
                yolo_score=0.7,
                verification=None,
            )
            _write_video_labels(labels_path, [("FallForwardS1.avi", 1)])
            repository = EventRepository(db_path).initialize()
            repository.create_candidate_event(
                event_id="event1",
                camera_id="cam1",
                source_uri="FallForwardS1.avi",
                clip_path="event1.mp4",
                metadata_path=str(event_dir / "cam1" / "20260616" / "event1.json"),
                candidate={
                    "candidate_id": "candidate_event1",
                    "source_uri": "FallForwardS1.avi",
                    "timestamp_ms": 1000,
                    "score": 0.7,
                },
                yolo_score=0.7,
            )
            repository.enqueue_vlm_job("event1")
            repository.complete_vlm_job(
                "vlm_event1",
                verification={
                    "result": "confirmed_fall",
                    "confidence": 0.95,
                    "reason": "VLM confirmed the fall.",
                },
                final_status="confirmed_fall",
            )

            report = build_caucafall_report(event_dir, labels_path, db_path)

        self.assertEqual(report["vlm"]["confirmed_fall"], 1)
        self.assertEqual(report["vlm"]["pending_candidates"], 0)
        self.assertEqual(report["vlm"]["average_confidence"], 0.95)
        detail = report["details"][0]
        self.assertEqual(detail["final_status"], "confirmed_fall")
        self.assertEqual(detail["prediction"], "fall")
        self.assertEqual(detail["outcome"], "TP")
        self.assertEqual(detail["vlm_reason"], "VLM confirmed the fall.")

    def test_render_markdown_uses_chinese_labels_and_reasons(self):
        report = {
            "event_dir": "events",
            "labels_path": "labels.csv",
            "videos": {"total": 1, "fall": 1, "nofall": 0},
            "yolo": {"candidates": 1, "average_score": 0.8},
            "vlm": {
                "confirmed_fall": 1,
                "need_human_review": 0,
                "rejected": 0,
                "pending_candidates": 0,
                "average_confidence": 0.9,
            },
            "metrics": {
                "true_positives": 1,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
            "details": [
                {
                    "source_uri": "FallForwardS1.avi",
                    "true_label": "fall",
                    "prediction": "fall",
                    "final_status": "confirmed_fall",
                    "outcome": "TP",
                    "yolo_score": 0.8,
                    "vlm_confidence": 0.9,
                    "vlm_reason": "老人倒地。",
                }
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("# CAUCAFall 视频级评估报告", markdown)
        self.assertIn("确认摔倒", markdown)
        self.assertIn("老人倒地。", markdown)

    def test_write_outputs_creates_json_markdown_and_detail_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = {
                "event_dir": "events",
                "labels_path": "labels.csv",
                "videos": {"total": 1, "fall": 1, "nofall": 0},
                "yolo": {"candidates": 1, "average_score": 0.8},
                "vlm": {
                    "confirmed_fall": 1,
                    "need_human_review": 0,
                    "rejected": 0,
                    "pending_candidates": 0,
                    "average_confidence": 0.9,
                },
                "metrics": {
                    "true_positives": 1,
                    "false_positives": 0,
                    "true_negatives": 0,
                    "false_negatives": 0,
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                },
                "details": [
                    {
                        "source_uri": "FallForwardS1.avi",
                        "true_label": "fall",
                        "prediction": "fall",
                        "final_status": "confirmed_fall",
                        "outcome": "TP",
                        "yolo_score": 0.8,
                        "vlm_confidence": 0.9,
                        "vlm_reason": "老人倒地。",
                    }
                ],
            }

            paths = write_outputs(
                report,
                output_json=root / "report.json",
                output_md=root / "report.md",
                output_csv=root / "details.csv",
            )

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["markdown"]).exists())
            self.assertTrue(Path(paths["details_csv"]).exists())
            loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["metrics"]["true_positives"], 1)
            csv_text = Path(paths["details_csv"]).read_text(encoding="utf-8")
            self.assertIn("FallForwardS1.avi", csv_text)


def _write_video_labels(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_uri", "has_fall", "label"])
        writer.writeheader()
        for source_uri, has_fall in rows:
            writer.writerow(
                {
                    "source_uri": source_uri,
                    "has_fall": has_fall,
                    "label": "fall" if has_fall else "nofall",
                }
            )


def _write_event(path, source_uri, category, yolo_score, verification):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "event_id": path.stem,
        "camera_id": "cam1",
        "source_uri": source_uri,
        "category": category,
        "candidate": {
            "candidate_id": f"candidate_{path.stem}",
            "source_uri": source_uri,
            "timestamp_ms": 1000,
            "score": yolo_score,
        },
        "verification": verification,
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
