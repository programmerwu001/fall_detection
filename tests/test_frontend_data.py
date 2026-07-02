import json
import tempfile
import unittest
from pathlib import Path

from services.event_repository import EventRepository
from services.frontend_data import (
    PUBLIC_EVENT_FIELDS,
    alerts,
    camera_dashboard,
    event_detail,
    evaluation_summary,
    fall_events,
    list_events,
    media_token_for_path,
    resolve_media_token,
    review_alerts,
    selected_config,
    showcase_cases,
)
import services.frontend_data as frontend_data


class FrontendDataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.event_dir = self.root / "events"
        self.config_path = self.root / "detection_config.json"
        self.preview_dir = self.root / "privacy_previews"
        self.confirmed_clip = self.event_dir / "confirmed.mp4"
        self.candidate_clip = self.event_dir / "candidate.mp4"
        self.review_clip = self.event_dir / "review.mp4"
        self.rejected_clip = self.event_dir / "rejected.mp4"
        for clip in (
            self.confirmed_clip,
            self.candidate_clip,
            self.review_clip,
            self.rejected_clip,
        ):
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"fake mp4")
        self._write_events()
        self.original_preview_dir = frontend_data.PRIVACY_PREVIEW_DIR
        frontend_data.PRIVACY_PREVIEW_DIR = self.preview_dir

    def tearDown(self):
        frontend_data.PRIVACY_PREVIEW_DIR = self.original_preview_dir
        self.temp_dir.cleanup()

    def test_list_events_reads_json_and_standardizes_rows(self):
        rows = list_events(self.event_dir, None)

        self.assertEqual(len(rows), 4)
        confirmed = next(row for row in rows if row["event_id"] == "event_confirmed")
        self.assertEqual(confirmed["display_status"], "high_risk")
        self.assertEqual(confirmed["status_label"], "高风险摔倒告警")
        self.assertEqual(confirmed["camera_id"], "file_cam_001")
        self.assertEqual(confirmed["area_label"], "模拟区域")
        self.assertEqual(confirmed["yolo_score"], 0.82)
        self.assertEqual(confirmed["vlm_label"], "高风险摔倒告警")
        self.assertEqual(confirmed["vlm_confidence"], 0.91)
        self.assertEqual(confirmed["risk_level"], "high_risk")
        self.assertTrue(set(confirmed).issubset(PUBLIC_EVENT_FIELDS))
        self.assertNotIn("clip_path", confirmed)
        self.assertNotIn("media_url", confirmed)
        self.assertNotIn("privacy_preview_path", confirmed)
        self.assertNotIn("source_uri", confirmed)
        self.assertNotIn("metadata_path", confirmed)
        self.assertNotIn("confirmed.mp4", json.dumps(confirmed, ensure_ascii=False))
        self.assertNotIn("confirmed_fall", json.dumps(confirmed, ensure_ascii=False))

    def test_sqlite_status_overrides_json_category(self):
        db_path = self.root / "records.db"
        repository = EventRepository(db_path).initialize()
        repository.create_candidate_event(
            event_id="event_candidate",
            camera_id="file_cam_001",
            source_uri="E:/source/candidate.mp4",
            clip_path=str(self.candidate_clip),
            metadata_path=str(self.event_dir / "event_candidate.json"),
            candidate={"score": 0.66},
            yolo_score=0.66,
            status="confirmed_fall",
        )

        self.assertEqual(
            [row["event_id"] for row in fall_events(self.event_dir, db_path)],
            ["event_candidate", "event_confirmed"],
        )
        dashboard = camera_dashboard(self.event_dir, db_path)
        cam = next(
            camera
            for camera in dashboard["cameras"]
            if camera["camera_id"] == "file_cam_001"
        )
        self.assertEqual(cam["risk_status"], "high_risk")

    def test_camera_dashboard_aggregates_cameras_and_highest_risk(self):
        dashboard = camera_dashboard(self.event_dir, None)

        self.assertEqual(dashboard["summary"]["camera_count"], 3)
        self.assertEqual(dashboard["summary"]["high_risk"], 1)
        self.assertEqual(dashboard["summary"]["low_risk"], 1)
        self.assertEqual(dashboard["summary"]["pending_detection"], 1)
        self.assertEqual(dashboard["summary"]["no_alarm"], 1)
        cam1 = next(
            camera
            for camera in dashboard["cameras"]
            if camera["camera_id"] == "file_cam_001"
        )
        self.assertEqual(cam1["risk_status"], "normal")
        self.assertIn("暂无实时画面", cam1["placeholder_text"])
        self.assertNotIn("stream_url", cam1)
        cam3 = next(
            camera
            for camera in dashboard["cameras"]
            if camera["camera_id"] == "file_cam_003"
        )
        self.assertEqual(cam3["risk_status"], "normal")

    def test_grouping_hides_rejected_from_clickable_lists(self):
        self.assertEqual(
            [row["event_id"] for row in fall_events(self.event_dir, None)],
            ["event_confirmed"],
        )
        self.assertEqual(
            {row["event_id"] for row in review_alerts(self.event_dir, None)},
            {"event_review"},
        )
        self.assertEqual(
            {row["event_id"] for row in alerts(self.event_dir, None)},
            {"event_confirmed", "event_review"},
        )
        self.assertNotIn(
            "event_rejected",
            {row["event_id"] for row in showcase_cases(self.event_dir, None)},
        )

    def test_legacy_json_only_events_are_read_only_not_pending_alerts(self):
        confirmed = next(
            row for row in list_events(self.event_dir, None)
            if row["event_id"] == "event_confirmed"
        )

        self.assertEqual(confirmed["risk_level"], "high_risk")
        self.assertEqual(confirmed["alert_status"], "legacy_read_only")
        self.assertFalse(confirmed["can_handle"])

    def test_evaluation_summary_counts_rejected(self):
        summary = evaluation_summary(self.event_dir, None)

        self.assertEqual(summary["displayed_cases"], 2)
        self.assertEqual(summary["high_risk"], 1)
        self.assertEqual(summary["low_risk"], 1)
        self.assertEqual(summary["pending_detection"], 1)
        self.assertEqual(summary["no_alarm"], 1)
        self.assertEqual(summary["yolo"]["candidates"], 4)
        self.assertEqual(summary["vlm"]["verified_events"], 3)
        self.assertFalse(summary["label_evaluation"]["available"])

    def test_evaluation_summary_uses_sqlite_only_events_for_yolo_and_vlm(self):
        db_path = self.root / "records.db"
        sqlite_only_clip = self.event_dir / "sqlite_only.mp4"
        sqlite_only_clip.write_bytes(b"fake mp4")
        repository = EventRepository(db_path).initialize()
        repository.create_candidate_event(
            event_id="sqlite_only",
            camera_id="file_cam_009",
            source_uri="E:/source/sqlite_only.mp4",
            clip_path=str(sqlite_only_clip),
            metadata_path=str(self.event_dir / "missing.json"),
            candidate={"score": 0.75},
            yolo_score=0.75,
        )
        repository.enqueue_vlm_job("sqlite_only", job_id="job_sqlite_only")
        repository.complete_vlm_job(
            job_id="job_sqlite_only",
            verification={
                "result": "confirmed_fall",
                "confidence": 0.8,
                "reason": "confirmed from sqlite",
            },
            final_status="confirmed_fall",
        )

        summary = evaluation_summary(self.event_dir, db_path)

        self.assertEqual(summary["displayed_cases"], 3)
        self.assertEqual(summary["high_risk"], 2)
        self.assertEqual(summary["yolo"]["candidates"], 5)
        self.assertEqual(summary["vlm"]["verified_events"], 4)
        self.assertEqual(summary["vlm"]["high_risk"], 2)

    def test_event_detail_returns_standard_sections(self):
        detail = event_detail(self.event_dir, "event_confirmed", None)

        self.assertEqual(detail["event"]["event_id"], "event_confirmed")
        self.assertEqual(detail["candidate"]["score"], 0.82)
        self.assertEqual(detail["verification"]["confidence"], 0.91)
        serialized = json.dumps(detail, ensure_ascii=False)
        self.assertNotIn("clip_path", serialized)
        self.assertNotIn("media_url", serialized)
        self.assertNotIn("privacy_preview_path", serialized)
        self.assertNotIn("source_uri", serialized)
        self.assertNotIn("metadata_path", serialized)
        self.assertNotIn("confirmed.mp4", serialized)
        self.assertNotIn("confirmed_fall", serialized)
        self.assertNotIn("need_human_review", serialized)
        self.assertEqual(
            detail["status_explanations"]["privacy_status"],
            "原始视频，尚未加密",
        )
        self.assertEqual(
            detail["status_explanations"]["integrity_status"],
            "尚未生成完整性哈希",
        )
        self.assertEqual(
            detail["status_explanations"]["retention_status"],
            "尚未生成留存清单",
        )

    def test_media_token_round_trip_rejects_paths_outside_allowed_root(self):
        token = media_token_for_path(self.confirmed_clip)
        self.assertEqual(
            resolve_media_token(token, self.event_dir),
            self.confirmed_clip.resolve(),
        )

        outside = self.root / "outside.mp4"
        outside.write_bytes(b"outside")
        outside_token = media_token_for_path(outside)
        with self.assertRaises(ValueError):
            resolve_media_token(outside_token, self.event_dir)

    def test_ready_privacy_preview_exposes_controlled_media_url_only(self):
        db_path = self.root / "records.db"
        preview = self.preview_dir / "event_candidate" / "privacy_preview.mp4"
        preview.parent.mkdir(parents=True)
        preview.write_bytes(b"preview")
        repository = EventRepository(db_path).initialize()
        repository.create_candidate_event(
            event_id="event_candidate",
            camera_id="file_cam_001",
            source_uri="E:/source/candidate.mp4",
            clip_path=str(self.candidate_clip),
            metadata_path=str(self.event_dir / "event_candidate.json"),
            candidate={"score": 0.66},
            yolo_score=0.66,
        )
        repository.record_event_decision(
            event_id="event_candidate",
            verification={"result": "need_human_review", "confidence": 0.5},
            final_status="need_human_review",
        )
        job = repository.lease_privacy_preview_job("privacy-worker", 60)
        repository.complete_privacy_preview_job(job["job_id"], str(preview))

        event = event_detail(self.event_dir, "event_candidate", db_path)["event"]
        serialized = json.dumps(event, ensure_ascii=False)

        self.assertEqual(event["privacy_preview_status"], "ready")
        self.assertTrue(event["privacy_preview_url"].startswith("/media/"))
        self.assertNotIn(str(self.candidate_clip), serialized)
        self.assertNotIn(str(preview), serialized)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("privacy_preview_path", event)

    def test_failed_privacy_preview_has_no_media_url_and_no_raw_fallback(self):
        db_path = self.root / "records.db"
        repository = EventRepository(db_path).initialize()
        repository.create_candidate_event(
            event_id="event_candidate",
            camera_id="file_cam_001",
            source_uri="E:/source/candidate.mp4",
            clip_path=str(self.candidate_clip),
            metadata_path=str(self.event_dir / "event_candidate.json"),
            candidate={"score": 0.66},
            yolo_score=0.66,
        )
        repository.record_event_decision(
            event_id="event_candidate",
            verification={"result": "need_human_review", "confidence": 0.5},
            final_status="need_human_review",
        )
        job = repository.lease_privacy_preview_job("privacy-worker", 60)
        repository.fail_privacy_preview_job(job["job_id"], "mask model failed", max_retries=1)

        event = event_detail(self.event_dir, "event_candidate", db_path)["event"]
        serialized = json.dumps(event, ensure_ascii=False)

        self.assertEqual(event["privacy_preview_status"], "failed")
        self.assertNotIn("privacy_preview_url", event)
        self.assertNotIn(str(self.candidate_clip), serialized)
        self.assertNotIn("candidate.mp4", serialized)

    def test_selected_config_filters_comment_fields(self):
        self.config_path.write_text(
            json.dumps(
                {
                    "_说明": "ignore me",
                    "参数说明": {"video_dir": "ignore me"},
                    "video_dir": "data/test_videos",
                    "output_dir": "data/events",
                    "nested": {
                        "_说明": "ignore nested",
                        "candidate_threshold": 0.4,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        config = selected_config(self.config_path)

        self.assertNotIn("_说明", config)
        self.assertNotIn("参数说明", config)
        self.assertNotIn("_说明", config["nested"])
        self.assertEqual(config["video_dir"], "data/test_videos")
        self.assertEqual(config["nested"]["candidate_threshold"], 0.4)

    def _write_events(self):
        events = [
            {
                "event_id": "event_confirmed",
                "category": "confirmed_fall",
                "camera_id": "file_cam_001",
                "source_uri": "E:/source/fall.mp4",
                "clip_path": str(self.confirmed_clip),
                "duration_ms": 6000,
                "created_at": "2026-06-09T22:00:00",
                "candidate": {"score": 0.82},
                "verification": {
                    "result": "confirmed_fall",
                    "confidence": 0.91,
                    "reason": "person is lying on the floor",
                    "visible_evidence": ["body close to floor"],
                },
                "privacy_status": "raw_unprotected",
                "integrity_status": "not_hashed",
                "retention_status": "pending_manifest",
            },
            {
                "event_id": "event_candidate",
                "category": "candidates",
                "camera_id": "file_cam_001",
                "source_uri": "E:/source/candidate.mp4",
                "clip_path": str(self.candidate_clip),
                "duration_ms": 5000,
                "created_at": "2026-06-09T22:05:00",
                "candidate": {"score": 0.66},
            },
            {
                "event_id": "event_review",
                "category": "need_human_review",
                "camera_id": "file_cam_002",
                "source_uri": "E:/source/review.mp4",
                "clip_path": str(self.review_clip),
                "duration_ms": 5000,
                "created_at": "2026-06-09T22:10:00",
                "candidate": {"score": 0.71},
                "verification": {
                    "result": "need_human_review",
                    "confidence": 0.45,
                    "reason": "unclear body posture",
                },
            },
            {
                "event_id": "event_rejected",
                "category": "rejected",
                "camera_id": "file_cam_003",
                "source_uri": "E:/source/rejected.mp4",
                "clip_path": str(self.rejected_clip),
                "duration_ms": 4000,
                "created_at": "2026-06-09T22:15:00",
                "candidate": {"score": 0.58},
                "verification": {
                    "result": "rejected",
                    "confidence": 0.84,
                    "reason": "person is sitting",
                },
            },
        ]
        for event in events:
            path = self.event_dir / f"{event['event_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(event), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
