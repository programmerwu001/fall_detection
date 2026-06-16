import tempfile
import unittest
from pathlib import Path

from app import create_api_response, guess_content_type, route_static_path


class AppHelpersTest(unittest.TestCase):
    def test_create_api_response_returns_json_bytes(self):
        status, headers, body = create_api_response({"ok": True})

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(body, b'{\n  "ok": true\n}')

    def test_guess_content_type(self):
        self.assertEqual(
            guess_content_type("index.html"),
            "text/html; charset=utf-8",
        )
        self.assertEqual(
            guess_content_type("styles.css"),
            "text/css; charset=utf-8",
        )
        self.assertEqual(
            guess_content_type("app.js"),
            "application/javascript; charset=utf-8",
        )
        self.assertEqual(guess_content_type("event.mp4"), "video/mp4")

    def test_route_static_path_maps_root_to_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir)
            index = static_dir / "index.html"
            index.write_text("<html></html>", encoding="utf-8")

            self.assertEqual(route_static_path("/", static_dir), index.resolve())

    def test_route_static_path_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir)

            with self.assertRaises(ValueError):
                route_static_path("/static/../secret.txt", static_dir)


if __name__ == "__main__":
    unittest.main()
