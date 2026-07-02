import tempfile
import unittest
from pathlib import Path

from scripts.reset_test_data import reset_test_data


class ResetTestDataTest(unittest.TestCase):
    def test_reset_removes_database_sidecars_and_event_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "data" / "events"
            nested_dir = event_dir / "camera_1"
            nested_dir.mkdir(parents=True)
            (nested_dir / "event_1.json").write_text("{}", encoding="utf-8")
            (event_dir / "event_1.mp4").write_bytes(b"video")

            db_path = root / "data" / "records.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
                path.write_bytes(b"db")

            result = reset_test_data(
                event_dir=event_dir,
                db_path=db_path,
                workspace_root=root,
                dry_run=False,
            )

            self.assertFalse(db_path.exists())
            self.assertFalse(Path(str(db_path) + "-wal").exists())
            self.assertFalse(Path(str(db_path) + "-shm").exists())
            self.assertTrue(event_dir.exists())
            self.assertEqual(list(event_dir.iterdir()), [])
            self.assertEqual(result["mode"], "deleted")
            self.assertEqual(len(result["db_files"]), 3)
            self.assertEqual(result["event_entries"], 2)

    def test_reset_removes_private_and_disabled_debug_event_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "data" / "events"
            private_event_dir = root / "data" / "private_events"
            disabled_debug_event_dir = root / "data" / "debug_events_disabled"
            for directory in (event_dir, private_event_dir, disabled_debug_event_dir):
                nested_dir = directory / "camera_1"
                nested_dir.mkdir(parents=True)
                (nested_dir / "event_1.mp4").write_bytes(b"video")

            db_path = root / "data" / "records.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_bytes(b"db")

            result = reset_test_data(
                event_dir=event_dir,
                private_event_dir=private_event_dir,
                disabled_debug_event_dir=disabled_debug_event_dir,
                db_path=db_path,
                workspace_root=root,
                dry_run=False,
            )

            self.assertEqual(list(event_dir.iterdir()), [])
            self.assertEqual(list(private_event_dir.iterdir()), [])
            self.assertEqual(list(disabled_debug_event_dir.iterdir()), [])
            self.assertEqual(result["event_entries"], 1)
            self.assertEqual(result["private_event_entries"], 1)
            self.assertEqual(result["disabled_debug_event_entries"], 1)

    def test_dry_run_reports_targets_without_deleting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            event_dir = root / "data" / "events"
            event_dir.mkdir(parents=True)
            event_file = event_dir / "event_1.json"
            event_file.write_text("{}", encoding="utf-8")
            db_path = root / "data" / "records.db"
            db_path.write_bytes(b"db")

            result = reset_test_data(
                event_dir=event_dir,
                db_path=db_path,
                workspace_root=root,
                dry_run=True,
            )

            self.assertTrue(event_file.exists())
            self.assertTrue(db_path.exists())
            self.assertEqual(result["mode"], "dry-run")
            self.assertEqual(len(result["db_files"]), 1)
            self.assertEqual(result["event_entries"], 1)

    def test_reset_rejects_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as outside:
                root = Path(workspace)
                event_dir = Path(outside) / "events"
                db_path = root / "data" / "records.db"

                with self.assertRaises(ValueError):
                    reset_test_data(
                        event_dir=event_dir,
                        db_path=db_path,
                        workspace_root=root,
                        dry_run=True,
                    )


if __name__ == "__main__":
    unittest.main()
