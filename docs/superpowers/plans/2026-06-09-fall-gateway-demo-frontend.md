# Fall Gateway Demo Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, read-only frontend that lets a mentor demonstrate the Fall Edge Gateway project, including project value, technical chain, runtime outputs, typical events, and metrics.

**Architecture:** Use `app.py` as a small standard-library HTTP server and keep the frontend as static files under `static/`. Put file-system data loading and normalization in a focused service module so it can be unit tested without running a browser.

**Tech Stack:** Python standard library `http.server`, existing project services, SQLite via existing repository code, static HTML/CSS/JavaScript, `unittest`.

---

## Scope And Constraints

- Implement the design in `docs/superpowers/specs/2026-06-09-fall-gateway-demo-frontend-design.md`.
- Keep the frontend read-only.
- Do not start YOLO or VLM jobs from the web UI.
- Do not modify `run_gateway.py`, `run_vlm_worker.py`, or detection pipeline behavior.
- Use existing event JSON files, `data/records.db`, config JSON, and metrics helpers as data sources.
- Do not introduce React, Vue, Vite, Flask, FastAPI, or other new runtime dependencies.
- Preserve unrelated worktree changes.

## Files

Create:

- `services/frontend_data.py`
  Loads event metadata, normalizes event rows/details, builds overview payloads, reads selected config, and safely resolves media paths.

- `tests/test_frontend_data.py`
  Unit tests for normalization, sorting, empty states, media-path safety, and overview counts.

- `tests/test_app.py`
  Unit tests for HTTP route behavior using temporary data directories where practical.

- `static/index.html`
  Single-page frontend shell with six sections.

- `static/styles.css`
  Presentation-focused styling for the demo platform.

- `static/app.js`
  Fetches API data, renders all sections, filters typical cases, and updates event detail.

Modify:

- `app.py`
  Replace the empty file with a local HTTP server that serves static files, JSON API endpoints, and safe media files.

- `README.md`
  Add a short "前端演示" section with the command to start the frontend and the local URL.

## API Contract

Implement these routes:

```text
GET /
GET /static/<file>
GET /api/overview
GET /api/events
GET /api/events/<event_id>
GET /api/metrics
GET /api/config
GET /media/<encoded_path>
```

Use URL-safe Base64 for media paths:

- The API returns `media_url` like `/media/<token>`.
- The token encodes the absolute clip path.
- The server decodes the token and only serves the file if it is under the configured event directory.

## Task 1: Data Normalization Service

**Files:**

- Create: `services/frontend_data.py`
- Create: `tests/test_frontend_data.py`

- [ ] **Step 1: Write failing tests for event loading and normalization**

Create `tests/test_frontend_data.py` with these tests:

```python
import json
import tempfile
import unittest
from pathlib import Path

from services.frontend_data import (
    build_overview,
    event_detail,
    list_events,
    media_token_for_path,
    resolve_media_token,
    selected_config,
)


class FrontendDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.event_dir = self.root / "data" / "events"
        self.event_dir.mkdir(parents=True)
        self.config_path = self.root / "configs" / "detection_config.json"
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(
            json.dumps(
                {
                    "_说明": "ignored",
                    "video_dir": "videos",
                    "output_dir": str(self.event_dir),
                    "candidate_threshold": 0.4,
                    "skip_vlm": False,
                }
            ),
            encoding="utf-8",
        )
        self.clip_path = self.event_dir / "confirmed_fall" / "file_cam_001" / "event_a.mp4"
        self.clip_path.parent.mkdir(parents=True)
        self.clip_path.write_bytes(b"fake video")
        self.metadata_path = self.clip_path.with_suffix(".json")
        self.metadata_path.write_text(
            json.dumps(
                {
                    "event_id": "event_a",
                    "category": "confirmed_fall",
                    "camera_id": "file_cam_001",
                    "source_uri": "E:/source/video_a.mp4",
                    "clip_path": str(self.clip_path),
                    "duration_ms": 6000,
                    "frame_count": 90,
                    "fps": 15,
                    "created_at": "2026-06-09T22:00:00",
                    "candidate": {
                        "score": 0.82,
                        "timestamp_ms": 12000
                    },
                    "verification": {
                        "result": "confirmed_fall",
                        "confidence": 0.91,
                        "reason": "person is lying on the floor",
                        "visible_evidence": ["body close to floor"]
                    },
                    "privacy_status": "raw_unprotected",
                    "integrity_status": "not_hashed",
                    "retention_status": "pending_manifest"
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_events_normalizes_rows(self):
        events = list_events(self.event_dir)
        self.assertEqual(len(events), 1)
        row = events[0]
        self.assertEqual(row["event_id"], "event_a")
        self.assertEqual(row["category_label"], "确认摔倒")
        self.assertEqual(row["source_name"], "video_a.mp4")
        self.assertEqual(row["yolo_score"], 0.82)
        self.assertEqual(row["vlm_confidence"], 0.91)
        self.assertEqual(row["duration_seconds"], 6.0)
        self.assertTrue(row["media_url"].startswith("/media/"))

    def test_build_overview_counts_events(self):
        overview = build_overview(self.event_dir, queue_db_path=None)
        self.assertTrue(overview["has_data"])
        self.assertEqual(overview["counts"]["total_events"], 1)
        self.assertEqual(overview["counts"]["confirmed_fall"], 1)
        self.assertEqual(overview["counts"]["cameras"], 1)
        self.assertEqual(overview["counts"]["sources"], 1)
        self.assertEqual(len(overview["recent_events"]), 1)
        self.assertEqual(len(overview["showcase_events"]), 1)

    def test_event_detail_includes_status_explanations(self):
        detail = event_detail(self.event_dir, "event_a")
        self.assertEqual(detail["event"]["event_id"], "event_a")
        self.assertEqual(detail["candidate"]["score"], 0.82)
        self.assertEqual(detail["verification"]["result"], "confirmed_fall")
        self.assertEqual(
            detail["status_explanations"]["privacy_status"],
            "原始视频，尚未加密",
        )

    def test_media_token_resolves_only_under_event_dir(self):
        token = media_token_for_path(self.clip_path)
        resolved = resolve_media_token(token, allowed_root=self.event_dir)
        self.assertEqual(resolved, self.clip_path.resolve())

    def test_media_token_rejects_path_outside_event_dir(self):
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"x")
        token = media_token_for_path(outside)
        with self.assertRaises(ValueError):
            resolve_media_token(token, allowed_root=self.event_dir)

    def test_selected_config_ignores_comment_keys(self):
        config = selected_config(self.config_path)
        self.assertNotIn("_说明", config)
        self.assertEqual(config["candidate_threshold"], 0.4)
        self.assertFalse(config["skip_vlm"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run:

```powershell
python -m unittest tests.test_frontend_data
```

Expected result:

```text
ImportError: cannot import name ...
```

or:

```text
ModuleNotFoundError: No module named 'services.frontend_data'
```

- [ ] **Step 3: Implement `services/frontend_data.py`**

Create `services/frontend_data.py` with this implementation:

```python
"""Read-only data helpers for the local frontend demo."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from services.event_repository import EventRepository
from services.metrics_report import build_metrics_report, load_event_metadata


CATEGORY_LABELS = {
    "confirmed_fall": "确认摔倒",
    "need_human_review": "需人工复核",
    "candidates": "候选事件",
    "rejected": "已拒绝",
}

CATEGORY_PRIORITY = {
    "confirmed_fall": 0,
    "need_human_review": 1,
    "candidates": 2,
    "rejected": 3,
}

STATUS_EXPLANATIONS = {
    "privacy_status": {
        "raw_unprotected": "原始视频，尚未加密",
    },
    "integrity_status": {
        "not_hashed": "尚未生成完整性哈希",
    },
    "retention_status": {
        "pending_manifest": "尚未生成留存清单",
    },
}

CONFIG_COMMENT_KEYS = {"_注释", "_说明", "_comment", "comment", "参数说明", "使用说明"}


def list_events(event_dir: str | Path) -> List[Dict[str, Any]]:
    """Return normalized event rows sorted for external demos."""
    rows = [_normalize_event(item) for item in load_event_metadata(Path(event_dir))]
    return sorted(rows, key=_event_sort_key)


def event_detail(event_dir: str | Path, event_id: str) -> Dict[str, Any]:
    """Return one normalized event detail payload."""
    event_id = str(event_id).strip()
    if not event_id:
        raise ValueError("event_id must be provided")

    for raw in load_event_metadata(Path(event_dir)):
        if str(raw.get("event_id") or "") != event_id:
            continue
        normalized = _normalize_event(raw)
        return {
            "event": normalized,
            "candidate": raw.get("candidate") if isinstance(raw.get("candidate"), dict) else {},
            "verification": raw.get("verification") if isinstance(raw.get("verification"), dict) else {},
            "status_explanations": _status_explanations(raw),
        }
    raise KeyError(event_id)


def build_overview(
    event_dir: str | Path,
    queue_db_path: Optional[str | Path],
) -> Dict[str, Any]:
    """Return data for the run overview page."""
    events = list_events(event_dir)
    categories = _count_by(events, "category")
    cameras = {row["camera_id"] for row in events if row.get("camera_id")}
    sources = {row["source_uri"] for row in events if row.get("source_uri")}
    return {
        "has_data": bool(events),
        "counts": {
            "total_events": len(events),
            "confirmed_fall": categories.get("confirmed_fall", 0),
            "need_human_review": categories.get("need_human_review", 0),
            "candidates": categories.get("candidates", 0),
            "rejected": categories.get("rejected", 0),
            "cameras": len(cameras),
            "sources": len(sources),
        },
        "queue": queue_status(queue_db_path),
        "recent_events": _recent_events(events, limit=5),
        "showcase_events": _showcase_events(events, limit=5),
    }


def queue_status(queue_db_path: Optional[str | Path]) -> Dict[str, Any]:
    """Return VLM queue status with a stable shape."""
    empty_jobs = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    if queue_db_path is None:
        return {
            "available": False,
            "jobs": empty_jobs,
            "reason": "queue db was not requested",
        }
    path = Path(queue_db_path)
    if not path.exists():
        return {
            "available": False,
            "jobs": empty_jobs,
            "reason": "queue db does not exist",
        }
    try:
        jobs = EventRepository(path).get_queue_stats()
    except Exception as exc:
        return {
            "available": False,
            "jobs": empty_jobs,
            "reason": str(exc),
        }
    return {
        "available": True,
        "jobs": {key: int(jobs.get(key, 0)) for key in empty_jobs},
        "reason": "",
    }


def metrics_payload(event_dir: str | Path, queue_db_path: Optional[str | Path]) -> Dict[str, Any]:
    """Build metrics in memory for the frontend."""
    return build_metrics_report(event_dir=Path(event_dir), queue_db_path=queue_db_path)


def selected_config(config_path: str | Path) -> Dict[str, Any]:
    """Return public config values for display."""
    path = Path(config_path)
    if not path.exists():
        return {"available": False, "reason": "config file does not exist"}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": str(exc)}
    if not isinstance(loaded, dict):
        return {"available": False, "reason": "config file must contain an object"}
    return {
        key: value
        for key, value in loaded.items()
        if not _is_config_comment_key(key)
    }


def media_token_for_path(path: str | Path) -> str:
    """Encode a file path for use in a media URL."""
    raw = str(Path(path).resolve()).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def resolve_media_token(token: str, allowed_root: str | Path) -> Path:
    """Resolve a media token and ensure it stays under allowed_root."""
    token = str(token).strip()
    if not token:
        raise ValueError("media token must be provided")
    padded = token + ("=" * (-len(token) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("invalid media token") from exc
    path = Path(decoded).resolve()
    root = Path(allowed_root).resolve()
    if path != root and root not in path.parents:
        raise ValueError("media path is outside allowed root")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    candidate = event.get("candidate") if isinstance(event.get("candidate"), dict) else {}
    verification = event.get("verification") if isinstance(event.get("verification"), dict) else {}
    category = str(event.get("category") or "unknown")
    source_uri = str(event.get("source_uri") or candidate.get("source_uri") or "")
    clip_path = str(event.get("clip_path") or "")
    return {
        "event_id": str(event.get("event_id") or ""),
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "camera_id": str(event.get("camera_id") or candidate.get("camera_id") or ""),
        "source_name": Path(source_uri).name if source_uri else "",
        "source_uri": source_uri,
        "clip_path": clip_path,
        "media_url": f"/media/{media_token_for_path(clip_path)}" if clip_path else "",
        "yolo_score": _safe_float(candidate.get("score")),
        "vlm_result": str(verification.get("result") or ""),
        "vlm_confidence": _safe_float(verification.get("confidence")),
        "vlm_reason": str(verification.get("reason") or ""),
        "visible_evidence": verification.get("visible_evidence")
        if isinstance(verification.get("visible_evidence"), list)
        else [],
        "duration_seconds": _duration_seconds(event.get("duration_ms")),
        "frame_count": _safe_int(event.get("frame_count")),
        "fps": _safe_float(event.get("fps")),
        "created_at": str(event.get("created_at") or ""),
        "privacy_status": str(event.get("privacy_status") or ""),
        "integrity_status": str(event.get("integrity_status") or ""),
        "retention_status": str(event.get("retention_status") or ""),
        "metadata_path": str(event.get("metadata_path") or ""),
    }


def _status_explanations(event: Dict[str, Any]) -> Dict[str, str]:
    return {
        "privacy_status": STATUS_EXPLANATIONS["privacy_status"].get(
            str(event.get("privacy_status") or ""),
            str(event.get("privacy_status") or "未记录"),
        ),
        "integrity_status": STATUS_EXPLANATIONS["integrity_status"].get(
            str(event.get("integrity_status") or ""),
            str(event.get("integrity_status") or "未记录"),
        ),
        "retention_status": STATUS_EXPLANATIONS["retention_status"].get(
            str(event.get("retention_status") or ""),
            str(event.get("retention_status") or "未记录"),
        ),
    }


def _event_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        CATEGORY_PRIORITY.get(str(row.get("category") or ""), 99),
        str(row.get("created_at") or ""),
        str(row.get("event_id") or ""),
    )


def _recent_events(events: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return sorted(events, key=lambda row: str(row.get("created_at") or ""), reverse=True)[:limit]


def _showcase_events(events: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return sorted(events, key=_event_sort_key)[:limit]


def _count_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _duration_seconds(value: Any) -> Optional[float]:
    duration_ms = _safe_float(value)
    if duration_ms is None:
        return None
    return round(duration_ms / 1000.0, 3)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _is_config_comment_key(key: str) -> bool:
    return key in CONFIG_COMMENT_KEYS or key.startswith("_")
```

- [ ] **Step 4: Run data service tests**

Run:

```powershell
python -m unittest tests.test_frontend_data
```

Expected result:

```text
Ran 6 tests

OK
```

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add services/frontend_data.py tests/test_frontend_data.py
git commit -m "feat: add frontend data service"
```

## Task 2: HTTP Server And API Routes

**Files:**

- Modify: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write failing tests for API route helpers**

Create `tests/test_app.py` with these tests:

```python
import json
import tempfile
import unittest
from pathlib import Path

from app import create_api_response, guess_content_type, route_static_path


class AppRouteTests(unittest.TestCase):
    def test_create_api_response_returns_json_bytes(self):
        status, headers, body = create_api_response({"ok": True})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(json.loads(body.decode("utf-8")), {"ok": True})

    def test_guess_content_type_handles_static_assets(self):
        self.assertEqual(guess_content_type("index.html"), "text/html; charset=utf-8")
        self.assertEqual(guess_content_type("styles.css"), "text/css; charset=utf-8")
        self.assertEqual(guess_content_type("app.js"), "application/javascript; charset=utf-8")
        self.assertEqual(guess_content_type("event.mp4"), "video/mp4")

    def test_route_static_path_resolves_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp)
            index = static_dir / "index.html"
            index.write_text("<html></html>", encoding="utf-8")
            resolved = route_static_path("/", static_dir)
            self.assertEqual(resolved, index.resolve())

    def test_route_static_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp)
            with self.assertRaises(ValueError):
                route_static_path("/static/../secret.txt", static_dir)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run:

```powershell
python -m unittest tests.test_app
```

Expected result:

```text
ImportError: cannot import name ...
```

- [ ] **Step 3: Implement `app.py`**

Replace the empty `app.py` with this implementation:

```python
"""Local read-only frontend server for Fall Edge Gateway."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import unquote, urlparse

from config import DB_PATH, EVENT_DIR
from run_gateway import DEFAULT_CONFIG_PATH
from services.frontend_data import (
    build_overview,
    event_detail,
    list_events,
    metrics_payload,
    resolve_media_token,
    selected_config,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), FrontendRequestHandler)
    print("Fall Edge Gateway frontend: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")
    finally:
        server.server_close()
    return 0


class FrontendRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/overview":
                self._send_api(build_overview(EVENT_DIR, DB_PATH))
            elif path == "/api/events":
                self._send_api({"events": list_events(EVENT_DIR)})
            elif path.startswith("/api/events/"):
                event_id = unquote(path.removeprefix("/api/events/"))
                self._send_api(event_detail(EVENT_DIR, event_id))
            elif path == "/api/metrics":
                self._send_api(metrics_payload(EVENT_DIR, DB_PATH))
            elif path == "/api/config":
                self._send_api(selected_config(DEFAULT_CONFIG_PATH))
            elif path.startswith("/media/"):
                token = path.removeprefix("/media/")
                self._send_file(resolve_media_token(token, allowed_root=EVENT_DIR))
            else:
                self._send_file(route_static_path(path, STATIC_DIR))
        except KeyError:
            self._send_api({"error": "not found"}, status=404)
        except FileNotFoundError:
            self._send_api({"error": "not found"}, status=404)
        except ValueError as exc:
            self._send_api({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_api({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_api(self, payload: Dict[str, Any], status: int = 200) -> None:
        response_status, headers, body = create_api_response(payload, status=status)
        self.send_response(response_status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", guess_content_type(path.name))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_api_response(payload: Dict[str, Any], status: int = 200) -> Tuple[int, Dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return (
        status,
        {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
        },
        body,
    )


def route_static_path(request_path: str, static_dir: Path) -> Path:
    path = "/" if not request_path else request_path
    if path == "/":
        relative = Path("index.html")
    elif path.startswith("/static/"):
        relative = Path(unquote(path.removeprefix("/static/")))
    else:
        relative = Path("index.html")

    root = static_dir.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("static path is outside static directory")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def guess_content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".ogg": "video/ogg",
    }.get(suffix, "application/octet-stream")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run app route tests**

Run:

```powershell
python -m unittest tests.test_app
```

Expected result:

```text
Ran 4 tests

OK
```

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add app.py tests/test_app.py
git commit -m "feat: add local frontend server"
```

## Task 3: Static Frontend Shell

**Files:**

- Create: `static/index.html`
- Create: `static/styles.css`
- Create: `static/app.js`

- [ ] **Step 1: Create `static/index.html`**

Create the HTML shell:

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>养老照护摔倒检测与可信事件留存演示平台</title>
    <link rel="stylesheet" href="/static/styles.css" />
  </head>
  <body>
    <header class="app-header">
      <div>
        <p class="eyebrow">Fall Edge Gateway</p>
        <h1>养老照护摔倒检测与可信事件留存演示平台</h1>
        <p class="subtitle">基于 YOLO 候选检测与 Video VLM 复核的养老场景风险事件识别原型</p>
      </div>
      <nav class="nav" aria-label="主导航">
        <a href="#overview">项目概览</a>
        <a href="#pain">场景痛点</a>
        <a href="#architecture">技术方案</a>
        <a href="#runtime">运行概览</a>
        <a href="#cases">典型案例</a>
        <a href="#metrics">评估结果</a>
      </nav>
    </header>

    <main>
      <section id="overview" class="section hero-section"></section>
      <section id="pain" class="section"></section>
      <section id="architecture" class="section"></section>
      <section id="runtime" class="section"></section>
      <section id="cases" class="section"></section>
      <section id="metrics" class="section"></section>
    </main>

    <script src="/static/app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Create `static/styles.css`**

Create focused styling:

```css
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --panel-soft: #eef3f2;
  --text: #172026;
  --muted: #5f6b73;
  --line: #d9e0e3;
  --accent: #0f766e;
  --accent-dark: #115e59;
  --warning: #b45309;
  --danger: #b91c1c;
  --radius: 8px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
}

.app-header {
  position: sticky;
  top: 0;
  z-index: 5;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 24px;
  align-items: center;
  padding: 18px 32px;
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid var(--line);
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--accent-dark);
  font-size: 13px;
  font-weight: 700;
}

h1 {
  margin: 0;
  font-size: 26px;
  line-height: 1.25;
}

h2 {
  margin: 0 0 16px;
  font-size: 24px;
}

h3 {
  margin: 0 0 8px;
  font-size: 17px;
}

.subtitle {
  margin: 8px 0 0;
  color: var(--muted);
}

.nav {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.nav a,
.filter-button {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  background: #fff;
  color: var(--text);
  text-decoration: none;
  font-size: 14px;
  cursor: pointer;
}

.nav a:hover,
.filter-button:hover,
.filter-button.active {
  border-color: var(--accent);
  color: var(--accent-dark);
}

.section {
  max-width: 1180px;
  margin: 0 auto;
  padding: 38px 32px;
}

.hero-section {
  padding-top: 48px;
}

.grid {
  display: grid;
  gap: 16px;
}

.grid-2 {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.grid-3 {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.grid-4 {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 18px;
}

.metric-card strong {
  display: block;
  font-size: 30px;
  line-height: 1.1;
}

.metric-card span {
  color: var(--muted);
  font-size: 14px;
}

.pipeline {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
}

.pipeline-step {
  min-height: 82px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px;
  background: var(--panel-soft);
  font-weight: 700;
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: var(--panel);
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
  font-size: 14px;
}

th {
  background: #f1f5f6;
  color: #334047;
  font-weight: 700;
}

.case-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
  gap: 18px;
}

.case-row {
  cursor: pointer;
}

.case-row:hover {
  background: #f6faf9;
}

.case-row.selected {
  background: #e8f5f2;
}

video {
  width: 100%;
  max-height: 420px;
  background: #101820;
  border-radius: var(--radius);
}

.muted {
  color: var(--muted);
}

.empty {
  padding: 18px;
  border: 1px dashed var(--line);
  border-radius: var(--radius);
  background: #fff;
  color: var(--muted);
}

.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 3px 8px;
  background: #e8f5f2;
  color: var(--accent-dark);
  font-size: 12px;
  font-weight: 700;
}

.badge.rejected {
  background: #fef2f2;
  color: var(--danger);
}

.badge.review {
  background: #fff7ed;
  color: var(--warning);
}

@media (max-width: 900px) {
  .app-header,
  .case-layout {
    grid-template-columns: 1fr;
  }

  .nav {
    justify-content: flex-start;
  }

  .grid-2,
  .grid-3,
  .grid-4,
  .pipeline {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 3: Create `static/app.js`**

Create the frontend logic:

```javascript
const state = {
  events: [],
  selectedEventId: null,
  activeFilter: "all",
};

const labels = {
  confirmed_fall: "确认摔倒",
  need_human_review: "需人工复核",
  candidates: "候选事件",
  rejected: "已拒绝",
};

document.addEventListener("DOMContentLoaded", () => {
  renderStaticSections();
  loadDynamicData();
});

async function loadDynamicData() {
  const [overview, eventsPayload, metrics] = await Promise.all([
    fetchJson("/api/overview"),
    fetchJson("/api/events"),
    fetchJson("/api/metrics"),
  ]);
  state.events = eventsPayload.events || [];
  state.selectedEventId = state.events[0]?.event_id || null;
  renderRuntime(overview);
  renderCases();
  renderMetrics(metrics);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function renderStaticSections() {
  document.querySelector("#overview").innerHTML = `
    <h2>项目概览</h2>
    <p class="subtitle">面向养老照护场景，自动发现疑似摔倒事件，并保存关键视频片段与事件元数据，为后续隐私保护和可信取证提供基础。</p>
    <div class="grid grid-3">
      ${capabilityCard("摔倒候选自动发现", "通过 YOLO 姿态/人体检测从视频流中筛选疑似风险事件。")}
      ${capabilityCard("VLM 复核降低误报", "对候选片段进行 Video VLM 复核，辅助区分摔倒、坐下、弯腰等情况。")}
      ${capabilityCard("事件片段与元数据留存", "保存事件前后片段、候选信息、复核结果和状态字段。")}
    </div>
    <div class="grid grid-2">
      <div class="card"><h3>已实现</h3><p>本地视频检测、YOLO 候选、事件缓存、异步 VLM 复核、事件保存、指标报告。</p></div>
      <div class="card"><h3>扩展方向</h3><p>视频隐私保护、哈希校验、防删除取证、授权查看。</p></div>
    </div>
  `;

  document.querySelector("#pain").innerHTML = `
    <h2>场景痛点</h2>
    <div class="grid grid-4">
      ${capabilityCard("摔倒发现不及时", "老人摔倒后，如果护理人员无法及时发现，可能错过处理窗口。")}
      ${capabilityCard("普通监控被动低效", "传统监控主要靠人工回放，不能主动筛选风险事件。")}
      ${capabilityCard("照护视频隐私敏感", "养老场景视频涉及个人隐私，需要后续支持受控查看与隐私保护。")}
      ${capabilityCard("事件证据可信度不足", "纠纷场景下，普通视频文件可能存在缺失、删除或难以证明完整性的问题。")}
    </div>
  `;

  document.querySelector("#architecture").innerHTML = `
    <h2>技术方案</h2>
    <div class="pipeline">
      ${["视频输入", "YOLO 检测", "摔倒候选", "事件缓存", "VLM 复核", "元数据与报告"].map(step => `<div class="pipeline-step">${step}</div>`).join("")}
    </div>
    <div class="grid grid-2">
      <div class="card"><h3>当前已实现链路</h3><p>本地视频目录输入、YOLO 候选生成、事件前后帧缓存、候选片段保存、SQLite 异步队列、VLM worker 复核、指标报告生成。</p></div>
      <div class="card"><h3>后续安全扩展方向</h3><p>事件视频加密、人体区域隐私保护、视频哈希校验、多节点备份、删除/篡改检测、授权查看。</p></div>
    </div>
  `;
}

function capabilityCard(title, body) {
  return `<article class="card"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(body)}</p></article>`;
}

function renderRuntime(overview) {
  const counts = overview.counts || {};
  const queue = overview.queue?.jobs || {};
  document.querySelector("#runtime").innerHTML = `
    <h2>运行概览</h2>
    ${overview.has_data ? "" : `<div class="empty">当前尚未生成检测输出，请先运行检测流程。</div>`}
    <div class="grid grid-4">
      ${metricCard("总事件数", counts.total_events)}
      ${metricCard("确认摔倒", counts.confirmed_fall)}
      ${metricCard("需人工复核", counts.need_human_review)}
      ${metricCard("候选事件", counts.candidates)}
      ${metricCard("已拒绝", counts.rejected)}
      ${metricCard("摄像头数量", counts.cameras)}
      ${metricCard("视频来源", counts.sources)}
      ${metricCard("VLM 完成任务", queue.done)}
    </div>
    <div class="grid grid-2">
      ${simpleEventTable("最近事件", overview.recent_events || [])}
      ${simpleEventTable("推荐展示事件", overview.showcase_events || [])}
    </div>
  `;
}

function metricCard(label, value) {
  return `<div class="card metric-card"><strong>${value ?? 0}</strong><span>${escapeHtml(label)}</span></div>`;
}

function simpleEventTable(title, events) {
  if (!events.length) {
    return `<div class="card"><h3>${escapeHtml(title)}</h3><p class="muted">暂无事件。</p></div>`;
  }
  return `
    <div class="card">
      <h3>${escapeHtml(title)}</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>类别</th><th>摄像头</th><th>YOLO</th></tr></thead>
          <tbody>
            ${events.map(event => `<tr><td>${badge(event.category)}</td><td>${escapeHtml(event.camera_id)}</td><td>${formatNumber(event.yolo_score)}</td></tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderCases() {
  document.querySelector("#cases").innerHTML = `
    <h2>典型案例</h2>
    <div class="filters">
      ${[
        ["all", "全部"],
        ["confirmed_fall", "确认摔倒"],
        ["need_human_review", "需人工复核"],
        ["candidates", "候选事件"],
        ["rejected", "已拒绝"],
      ].map(([value, label]) => `<button class="filter-button ${state.activeFilter === value ? "active" : ""}" data-filter="${value}">${label}</button>`).join("")}
    </div>
    <div class="case-layout">
      <div id="case-list"></div>
      <div id="case-detail" class="card"></div>
    </div>
  `;

  document.querySelectorAll(".filter-button").forEach(button => {
    button.addEventListener("click", () => {
      state.activeFilter = button.dataset.filter;
      const filtered = filteredEvents();
      state.selectedEventId = filtered[0]?.event_id || null;
      renderCases();
    });
  });

  renderCaseList();
  renderCaseDetail();
}

function filteredEvents() {
  if (state.activeFilter === "all") {
    return state.events;
  }
  return state.events.filter(event => event.category === state.activeFilter);
}

function renderCaseList() {
  const events = filteredEvents();
  const container = document.querySelector("#case-list");
  if (!events.length) {
    container.innerHTML = `<div class="empty">当前筛选条件下没有事件。</div>`;
    return;
  }
  container.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>类别</th><th>摄像头</th><th>来源</th><th>YOLO</th><th>VLM</th><th>时长</th></tr>
        </thead>
        <tbody>
          ${events.map(event => `
            <tr class="case-row ${event.event_id === state.selectedEventId ? "selected" : ""}" data-event-id="${escapeHtml(event.event_id)}">
              <td>${badge(event.category)}</td>
              <td>${escapeHtml(event.camera_id)}</td>
              <td>${escapeHtml(event.source_name)}</td>
              <td>${formatNumber(event.yolo_score)}</td>
              <td>${formatNumber(event.vlm_confidence)}</td>
              <td>${formatNumber(event.duration_seconds)}s</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
  document.querySelectorAll(".case-row").forEach(row => {
    row.addEventListener("click", () => {
      state.selectedEventId = row.dataset.eventId;
      renderCaseList();
      renderCaseDetail();
    });
  });
}

async function renderCaseDetail() {
  const container = document.querySelector("#case-detail");
  if (!state.selectedEventId) {
    container.innerHTML = `<p class="muted">请选择一个事件。</p>`;
    return;
  }
  const detail = await fetchJson(`/api/events/${encodeURIComponent(state.selectedEventId)}`);
  const event = detail.event || {};
  const verification = detail.verification || {};
  const explanations = detail.status_explanations || {};
  container.innerHTML = `
    <h3>${escapeHtml(event.event_id)}</h3>
    ${event.media_url ? `<video src="${event.media_url}" controls></video>` : `<div class="empty">该事件没有可播放片段。</div>`}
    <p><strong>类别：</strong>${badge(event.category)}</p>
    <p><strong>摄像头：</strong>${escapeHtml(event.camera_id)}</p>
    <p><strong>来源视频：</strong>${escapeHtml(event.source_name)}</p>
    <p><strong>YOLO 分数：</strong>${formatNumber(event.yolo_score)}</p>
    <p><strong>VLM 结果：</strong>${escapeHtml(verification.result || event.vlm_result || "未复核")}</p>
    <p><strong>VLM 置信度：</strong>${formatNumber(verification.confidence ?? event.vlm_confidence)}</p>
    <p><strong>判断理由：</strong>${escapeHtml(verification.reason || event.vlm_reason || "未记录")}</p>
    <p><strong>隐私状态：</strong>${escapeHtml(explanations.privacy_status || event.privacy_status)}</p>
    <p><strong>完整性状态：</strong>${escapeHtml(explanations.integrity_status || event.integrity_status)}</p>
    <p><strong>留存状态：</strong>${escapeHtml(explanations.retention_status || event.retention_status)}</p>
  `;
}

function renderMetrics(metrics) {
  const labelEvaluation = metrics.label_evaluation || {};
  const labelBlock = labelEvaluation.available
    ? `
      ${metricCard("Precision", labelEvaluation.precision)}
      ${metricCard("Recall", labelEvaluation.recall)}
      ${metricCard("F1", labelEvaluation.f1)}
      ${metricCard("1000ms 内准确率", labelEvaluation.start_time_accuracy?.within_1000ms)}
    `
    : `<div class="empty">当前未提供人工标注文件，因此 Precision、Recall、F1 和时间准确率尚未计算。</div>`;

  document.querySelector("#metrics").innerHTML = `
    <h2>评估结果</h2>
    <div class="grid grid-4">
      ${metricCard("事件总数", metrics.events?.total)}
      ${metricCard("片段总时长", metrics.clips?.duration_seconds?.total)}
      ${metricCard("平均片段时长", metrics.clips?.duration_seconds?.average)}
      ${metricCard("总保存帧数", metrics.clips?.frames?.total)}
      ${metricCard("YOLO 候选", metrics.yolo?.candidates)}
      ${metricCard("平均 YOLO 分数", metrics.yolo?.average_score)}
      ${metricCard("VLM 复核数", metrics.vlm?.verified_events)}
      ${metricCard("平均 VLM 置信度", metrics.vlm?.average_confidence)}
    </div>
    <h3>人工标注评估</h3>
    <div class="grid grid-4">${labelBlock}</div>
  `;
}

function badge(category) {
  const className = category === "rejected" ? "rejected" : category === "need_human_review" ? "review" : "";
  return `<span class="badge ${className}">${escapeHtml(labels[category] || category || "未知")}</span>`;
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const number = Number(value);
  if (Number.isNaN(number)) {
    return String(value);
  }
  return number.toFixed(3).replace(/\.?0+$/, "");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
```

- [ ] **Step 4: Start the server and manually inspect the page**

Run:

```powershell
python app.py
```

Expected terminal output:

```text
Fall Edge Gateway frontend: http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8000
```

Expected page behavior:

- Navigation shows all six sections.
- Static sections render without API data.
- Runtime, cases, and metrics sections load from API.
- Empty states render if no event data exists.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add static/index.html static/styles.css static/app.js
git commit -m "feat: add demo frontend UI"
```

## Task 4: Documentation And Full Verification

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Add README instructions**

Add this section to `README.md` after the "运行方式" section:

```markdown
## 前端演示

项目提供一个本地只读前端，用于展示项目概览、场景痛点、技术方案、运行概览、典型案例和评估结果。

启动前端：

```powershell
python app.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

前端读取现有的 `data/events`、`data/records.db`、`configs/detection_config.json` 和指标报告数据。第一版前端不会启动 YOLO 或 VLM 推理任务，也不会修改事件数据。
```

- [ ] **Step 2: Run focused unit tests**

Run:

```powershell
python -m unittest tests.test_frontend_data tests.test_app
```

Expected result:

```text
Ran 10 tests

OK
```

- [ ] **Step 3: Run existing test suite**

Run:

```powershell
python -m unittest discover -s tests
```

Expected result:

```text
OK
```

The exact number of tests may differ because this repository already has tests.

- [ ] **Step 4: Check API endpoints manually**

With the server running, open these URLs:

```text
http://127.0.0.1:8000/api/overview
http://127.0.0.1:8000/api/events
http://127.0.0.1:8000/api/metrics
http://127.0.0.1:8000/api/config
```

Expected behavior:

- Each endpoint returns JSON.
- `/api/overview` includes `counts`, `queue`, `recent_events`, and `showcase_events`.
- `/api/events` includes an `events` array.
- `/api/metrics` includes `events`, `clips`, `yolo`, `vlm`, `queue`, and `label_evaluation`.
- `/api/config` excludes comment keys such as `_说明` and `参数说明`.

- [ ] **Step 5: Verify browser behavior**

Open:

```text
http://127.0.0.1:8000
```

Check:

- 项目概览 renders project title and three capability cards.
- 场景痛点 renders four problem blocks.
- 技术方案 renders the six-step pipeline and separates implemented chain from future directions.
- 运行概览 renders runtime count cards and no Precision/Recall/F1.
- 典型案例 renders the event table, filters, selected detail, and a video element when media is available.
- 评估结果 renders metrics and shows the missing-label message when labels are unavailable.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add README.md
git commit -m "docs: document demo frontend"
```

## Final Review Checklist

Before handing the implementation back:

- [ ] `python -m unittest tests.test_frontend_data tests.test_app` passes.
- [ ] `python -m unittest discover -s tests` passes, or any failures are clearly unrelated and documented.
- [ ] `python app.py` starts the local frontend server.
- [ ] `/api/overview`, `/api/events`, `/api/metrics`, and `/api/config` return JSON.
- [ ] `/media/<token>` rejects paths outside `data/events`.
- [ ] The frontend has exactly these six sections: 项目概览, 场景痛点, 技术方案, 运行概览, 典型案例, 评估结果.
- [ ] No web UI control starts detection, VLM processing, encryption, deletion, or storage operations.
- [ ] The UI clearly labels privacy, hash, anti-deletion, and authorization as future extension directions.
- [ ] README contains the frontend start command and local URL.
