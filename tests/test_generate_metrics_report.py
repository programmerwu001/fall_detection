import json
import tempfile
import unittest
from pathlib import Path

import generate_metrics_report


class GenerateMetricsReportCliTest(unittest.TestCase):
    def test_main_writes_json_and_markdown_to_requested_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "events"
            metadata_path = event_dir / "candidates" / "cam1" / "20260609" / "event1.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "event_id": "event1",
                        "camera_id": "cam1",
                        "category": "candidates",
                        "source_uri": "source.mp4",
                        "duration_ms": 6000,
                        "frame_count": 90,
                        "candidate": {
                            "candidate_id": "candidate1",
                            "timestamp_ms": 1000,
                            "score": 0.82,
                        },
                        "verification": None,
                    }
                ),
                encoding="utf-8",
            )
            output_json = root / "report.json"
            output_md = root / "report.md"

            exit_code = generate_metrics_report.main(
                [
                    "--event-dir",
                    str(event_dir),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                    "--no-queue-db",
                ]
            )

            loaded = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(loaded["events"]["total"], 1)
        self.assertIn("Fall Edge Gateway Metrics Report", markdown)


if __name__ == "__main__":
    unittest.main()
