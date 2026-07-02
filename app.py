"""Local read-only frontend server for Fall Edge Gateway."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from config import (
    DB_PATH,
    DEFAULT_HIGH_RISK_REPEAT_SECONDS,
    DEFAULT_LOW_RISK_REPEAT_SECONDS,
    EVENT_DIR,
    PRIVACY_PREVIEW_DIR,
)
from services.alert_policy import HIGH_RISK, LOW_RISK
from services.event_repository import EventRepository
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
VIDEO_FILE_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".flv", ".webm"}


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


def parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    """解析单段 HTTP Range 头，返回闭区间字节范围。"""
    if file_size <= 0 or not range_header.startswith("bytes="):
        return None
    range_spec = range_header[len("bytes=") :].strip()
    if "," in range_spec:
        return None
    start_text, separator, end_text = range_spec.partition("-")
    if separator != "-":
        return None
    if not start_text:
        if not end_text.isdigit():
            return None
        suffix_length = int(end_text)
        if suffix_length <= 0:
            return None
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        if not start_text.isdigit() or (end_text and not end_text.isdigit()):
            return None
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
        if start >= file_size or start > end:
            return None
        end = min(end, file_size - 1)
    return start, end


def create_media_response(
    file_path: Path,
    range_header: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """创建支持字节范围请求的视频响应。"""
    file_size = file_path.stat().st_size
    headers = {
        "Content-Type": guess_content_type(file_path.name),
        "Accept-Ranges": "bytes",
    }
    if range_header:
        byte_range = parse_byte_range(range_header, file_size)
        if byte_range is None:
            headers["Content-Range"] = f"bytes */{file_size}"
            headers["Content-Length"] = "0"
            return 416, headers, b""
        start, end = byte_range
        with file_path.open("rb") as file:
            file.seek(start)
            body = file.read(end - start + 1)
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(len(body))
        return 206, headers, body

    body = file_path.read_bytes()
    headers["Content-Length"] = str(len(body))
    return 200, headers, body


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
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=403)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/events/") and path.endswith("/handle"):
                event_id = unquote(path[len("/api/events/") : -len("/handle")])
                handled_by = self.headers.get("X-User-Account") or "demo_caregiver"
                _configured_repository().mark_event_handled(
                    event_id=event_id,
                    handled_by=handled_by,
                )
                self._send_json(event_detail(EVENT_DIR, event_id, DB_PATH))
                return
            self._send_json({"error": "Not found"}, status=404)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=403)
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
        if path == "/api/reminders":
            reminders = [
                event
                for event in _configured_repository().claim_due_reminders()
                if event.get("alert_status") == "pending"
            ]
            self._send_json({"reminders": [_public_reminder(event) for event in reminders]})
            return
        if path == "/api/fall-events":
            self._send_json(
                {
                    "events": fall_events(EVENT_DIR, DB_PATH),
                    "empty_message": "当前暂无高风险摔倒告警。",
                }
            )
            return
        if path == "/api/review-alerts":
            self._send_json(
                {
                    "alerts": review_alerts(EVENT_DIR, DB_PATH),
                    "empty_message": "当前暂无低风险摔倒告警。",
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
        if file_path.suffix.lower() in VIDEO_FILE_SUFFIXES:
            raise PermissionError("Video files are not served")
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
        try:
            file_path = resolve_media_token(unquote(token), PRIVACY_PREVIEW_DIR)
        except ValueError as exc:
            raise PermissionError("Raw event media is not served by this endpoint") from exc
        if file_path.name != "privacy_preview.mp4":
            raise PermissionError("Raw event media is not served by this endpoint")
        status, headers, body = create_media_response(
            file_path,
            self.headers.get("Range"),
        )
        self._send_response(status, headers, body)

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


def _public_reminder(event: dict) -> dict:
    risk_level = str(event.get("risk_level") or "")
    high_repeat, low_repeat = _configured_repeat_seconds()
    return {
        "event_id": event.get("event_id"),
        "camera_id": event.get("camera_id"),
        "risk_level": risk_level,
        "alert_status": event.get("alert_status"),
        "reminder_count": event.get("reminder_count"),
        "last_notified_at": event.get("last_notified_at"),
        "next_remind_at": event.get("next_remind_at"),
        "repeat_seconds": high_repeat if risk_level == HIGH_RISK else low_repeat if risk_level == LOW_RISK else None,
    }


def _configured_repository() -> EventRepository:
    high_repeat, low_repeat = _configured_repeat_seconds()
    return EventRepository(
        DB_PATH,
        high_risk_repeat_seconds=high_repeat,
        low_risk_repeat_seconds=low_repeat,
    ).initialize()


def _configured_repeat_seconds() -> tuple[int, int]:
    config = selected_config(DEFAULT_CONFIG_PATH)
    return (
        _safe_int(config.get("high_risk_repeat_seconds"), DEFAULT_HIGH_RISK_REPEAT_SECONDS),
        _safe_int(config.get("low_risk_repeat_seconds"), DEFAULT_LOW_RISK_REPEAT_SECONDS),
    )


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
