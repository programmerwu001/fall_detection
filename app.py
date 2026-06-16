"""Local read-only frontend server for Fall Edge Gateway."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from config import DB_PATH, EVENT_DIR
from services.frontend_data import (
    alerts,
    camera_dashboard,
    event_detail,
    evaluation_summary,
    fall_events,
    resolve_media_token,
    review_alerts,
    selected_config,
    showcase_cases,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "detection_config.json"


def create_api_response(
    payload: dict,
    status: int = 200,
) -> tuple[int, dict[str, str], bytes]:
    """把 dict 编码成 JSON HTTP 响应。"""
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return (
        status,
        {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        },
        body,
    )


def route_static_path(request_path: str, static_dir: Path) -> Path:
    """把请求路径解析到 static 目录下，防止路径穿越。"""
    if request_path == "/":
        relative = "index.html"
    elif request_path.startswith("/static/"):
        relative = unquote(request_path[len("/static/") :])
    else:
        raise ValueError("Unsupported static path")

    if ".." in Path(relative).parts:
        raise ValueError("Static path traversal is not allowed")
    resolved = (static_dir / relative).resolve()
    root = static_dir.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Static path is outside static directory") from exc
    return resolved


def guess_content_type(filename: str) -> str:
    """根据后缀返回 Content-Type。"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


class FrontendRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/" or path.startswith("/static/"):
                self._send_static(path)
            elif path.startswith("/api/"):
                self._send_api(path)
            elif path.startswith("/media/"):
                self._send_media(path[len("/media/") :])
            else:
                self._send_json({"error": "Not found"}, status=404)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_api(self, path: str) -> None:
        if path == "/api/cameras":
            self._send_json(camera_dashboard(EVENT_DIR, DB_PATH))
            return
        if path == "/api/alerts":
            self._send_json({"alerts": alerts(EVENT_DIR, DB_PATH)})
            return
        if path == "/api/fall-events":
            self._send_json(
                {
                    "events": fall_events(EVENT_DIR, DB_PATH),
                    "empty_message": "当前暂无已确认摔倒事件。",
                }
            )
            return
        if path == "/api/review-alerts":
            self._send_json(
                {
                    "alerts": review_alerts(EVENT_DIR, DB_PATH),
                    "empty_message": "当前暂无待复核告警。",
                }
            )
            return
        if path == "/api/showcase":
            self._send_json({"cases": showcase_cases(EVENT_DIR, DB_PATH)})
            return
        if path.startswith("/api/events/"):
            event_id = unquote(path[len("/api/events/") :])
            self._send_json(event_detail(EVENT_DIR, event_id, DB_PATH))
            return
        if path == "/api/evaluation":
            self._send_json(evaluation_summary(EVENT_DIR, DB_PATH))
            return
        if path == "/api/config":
            self._send_json(selected_config(DEFAULT_CONFIG_PATH))
            return
        self._send_json({"error": "Not found"}, status=404)

    def _send_static(self, path: str) -> None:
        file_path = route_static_path(path, STATIC_DIR)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(str(file_path))
        body = file_path.read_bytes()
        self._send_response(
            200,
            {
                "Content-Type": guess_content_type(file_path.name),
                "Content-Length": str(len(body)),
            },
            body,
        )

    def _send_media(self, token: str) -> None:
        file_path = resolve_media_token(unquote(token), EVENT_DIR)
        body = file_path.read_bytes()
        self._send_response(
            200,
            {
                "Content-Type": guess_content_type(file_path.name),
                "Content-Length": str(len(body)),
            },
            body,
        )

    def _send_json(self, payload: dict, status: int = 200) -> None:
        response_status, headers, body = create_api_response(payload, status)
        self._send_response(response_status, headers, body)

    def _send_response(
        self,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), FrontendRequestHandler)
    print("Fall Edge Gateway frontend: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
