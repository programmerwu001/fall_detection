import tempfile
import http.client
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import app
from app import create_api_response, guess_content_type, route_static_path
from services.frontend_data import media_token_for_path


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

    def test_media_endpoint_blocks_raw_event_mp4(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_dir = Path(temp_dir) / "events"
            preview_dir = Path(temp_dir) / "privacy_previews"
            event_dir.mkdir()
            preview_dir.mkdir()
            clip = event_dir / "event.mp4"
            clip.write_bytes(b"0123456789")
            token = media_token_for_path(clip)
            original_event_dir = app.EVENT_DIR
            original_preview_dir = app.PRIVACY_PREVIEW_DIR
            app.EVENT_DIR = event_dir
            app.PRIVACY_PREVIEW_DIR = preview_dir
            server = ThreadingHTTPServer(("127.0.0.1", 0), app.FrontendRequestHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_port,
                timeout=5,
            )

            try:
                connection.request(
                    "GET",
                    f"/media/{token}",
                    headers={"Range": "bytes=2-5"},
                )
                response = connection.getresponse()
                body = response.read()

                self.assertEqual(response.status, 403)
                self.assertIn(b"Raw event media is not served", body)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=1)
                app.EVENT_DIR = original_event_dir
                app.PRIVACY_PREVIEW_DIR = original_preview_dir

    def test_media_endpoint_serves_privacy_preview_with_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_dir = Path(temp_dir) / "events"
            preview_dir = Path(temp_dir) / "privacy_previews"
            preview = preview_dir / "event1" / "privacy_preview.mp4"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"0123456789")
            token = media_token_for_path(preview, allowed_root=preview_dir)
            original_event_dir = app.EVENT_DIR
            original_preview_dir = app.PRIVACY_PREVIEW_DIR
            app.EVENT_DIR = event_dir
            app.PRIVACY_PREVIEW_DIR = preview_dir
            server = ThreadingHTTPServer(("127.0.0.1", 0), app.FrontendRequestHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_port,
                timeout=5,
            )

            try:
                connection.request(
                    "GET",
                    f"/media/{token}",
                    headers={"Range": "bytes=2-5"},
                )
                response = connection.getresponse()
                body = response.read()

                self.assertEqual(response.status, 206)
                self.assertEqual(body, b"2345")
                self.assertEqual(response.getheader("Content-Range"), "bytes 2-5/10")
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=1)
                app.EVENT_DIR = original_event_dir
                app.PRIVACY_PREVIEW_DIR = original_preview_dir

    def test_static_endpoint_blocks_video_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir)
            (static_dir / "event.mp4").write_bytes(b"raw video")
            original_static_dir = app.STATIC_DIR
            app.STATIC_DIR = static_dir
            server = ThreadingHTTPServer(("127.0.0.1", 0), app.FrontendRequestHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_port,
                timeout=5,
            )

            try:
                connection.request("GET", "/static/event.mp4")
                response = connection.getresponse()
                body = response.read()

                self.assertEqual(response.status, 403)
                self.assertIn(b"Video files are not served", body)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=1)
                app.STATIC_DIR = original_static_dir


if __name__ == "__main__":
    unittest.main()
