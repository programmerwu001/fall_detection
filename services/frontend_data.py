"""Read-only data helpers for the local fall monitoring frontend."""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from services import event_state
from services.metrics_report import build_metrics_report, load_event_metadata


SIMULATION_NOTE = (
    "当前原型使用本地视频模拟摄像头输入，因此本页暂不展示持续实时画面；"
    "接入 RTSP 或真实摄像头后，可替换为实时视频流。"
)
CAMERA_PLACEHOLDER_TEXT = "当前为模拟摄像头\n暂无实时画面\n最近告警可在回放中查看"

DISPLAY_ALERT_STATUSES = {"confirmed_fall", "need_human_review", "candidates"}
REVIEW_STATUSES = {"candidates", "need_human_review"}
COMMENT_KEYS = {"_说明", "参数说明"}

RISK_PRIORITY = {
    "confirmed_fall": 0,
    "need_human_review": 1,
    "candidates": 2,
    "normal": 3,
}

CATEGORY_LABELS = {
    "confirmed_fall": "已确认摔倒",
    "need_human_review": "待复核",
    "candidates": "疑似摔倒",
    "rejected": "已过滤误报",
    "normal": "正常",
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


def list_events(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """读取事件 JSON，合并 SQLite 状态，返回标准化事件列表。"""
    event_dir_path = Path(event_dir)
    sqlite_events = _load_sqlite_events(queue_db_path)
    metadata = load_event_metadata(event_dir_path)
    rows = [
        _standardize_event(event, sqlite_events.get(str(event.get("event_id") or "")))
        for event in metadata
    ]
    known_ids = {str(event.get("event_id") or "") for event in metadata}
    for event_id, sqlite_event in sqlite_events.items():
        if event_id not in known_ids:
            rows.append(_standardize_event(sqlite_event, sqlite_event))
    return _sort_events(rows)


def camera_dashboard(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> dict:
    """返回实时监测模块的 summary、cameras、latest_alerts、queue 和 simulation_note。"""
    events = list_events(event_dir, queue_db_path)
    cameras_by_id: Dict[str, List[dict]] = {}
    for event in events:
        camera_id = str(event.get("camera_id") or "").strip()
        if not camera_id:
            continue
        cameras_by_id.setdefault(camera_id, []).append(event)

    cameras = [
        _camera_card(camera_id, camera_events)
        for camera_id, camera_events in sorted(cameras_by_id.items())
    ]
    summary = {
        "camera_count": len(cameras),
        "risk_camera_count": sum(
            1 for camera in cameras if camera["risk_status"] != "normal"
        ),
        "confirmed_fall": _count_status(events, "confirmed_fall"),
        "review_alerts": _count_status(events, "need_human_review"),
        "candidate_alerts": _count_status(events, "candidates"),
        "rejected": _count_status(events, "rejected"),
    }
    return {
        "summary": summary,
        "cameras": cameras,
        "latest_alerts": alerts(event_dir, queue_db_path)[:8],
        "queue": _queue_summary(event_dir, queue_db_path),
        "simulation_note": SIMULATION_NOTE,
        "empty_message": "暂无摄像头事件数据",
    }


def alerts(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回告警中心列表，不包含 rejected。"""
    return [
        event
        for event in list_events(event_dir, queue_db_path)
        if event["display_status"] in DISPLAY_ALERT_STATUSES
    ]


def fall_events(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回已确认摔倒事件。"""
    return [
        event
        for event in list_events(event_dir, queue_db_path)
        if event["display_status"] == "confirmed_fall"
    ]


def review_alerts(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回 candidates 和 need_human_review 告警。"""
    return [
        event
        for event in list_events(event_dir, queue_db_path)
        if event["display_status"] in REVIEW_STATUSES
    ]


def showcase_cases(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回可点击展示案例，不包含 rejected。"""
    return alerts(event_dir, queue_db_path)


def evaluation_summary(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> dict:
    """返回案例回放与评估模块使用的指标摘要，包含 rejected 数量。"""
    events = list_events(event_dir, queue_db_path)
    report = build_metrics_report(event_dir, queue_db_path)
    return {
        "displayed_cases": sum(
            1 for event in events if event["display_status"] in DISPLAY_ALERT_STATUSES
        ),
        "confirmed_fall": _count_status(events, "confirmed_fall"),
        "review_alerts": _count_status(events, "need_human_review"),
        "candidate_alerts": _count_status(events, "candidates"),
        "rejected": _count_status(events, "rejected"),
        "yolo": _summarize_frontend_yolo(events),
        "vlm": _summarize_frontend_vlm(events),
        "label_evaluation": report.get("label_evaluation", {}),
        "queue": report.get("queue", {}),
    }


def event_detail(
    event_dir: str | Path,
    event_id: str,
    queue_db_path: str | Path | None = None,
) -> dict:
    """返回单个事件详情。"""
    for event in list_events(event_dir, queue_db_path):
        if event["event_id"] == event_id:
            return {
                "event": event,
                "candidate": event.get("candidate") or {},
                "verification": event.get("verification") or {},
                "status_explanations": _status_explanations(event),
            }
    raise KeyError(f"Event not found: {event_id}")


def selected_config(config_path: str | Path) -> dict:
    """读取适合前端展示的配置字段，过滤注释字段。"""
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        return {}
    return _filter_comment_fields(loaded)


def media_token_for_path(path: str | Path) -> str:
    """把文件路径编码为媒体 URL token。"""
    encoded = base64.urlsafe_b64encode(str(Path(path).resolve()).encode("utf-8"))
    return encoded.decode("ascii")


def resolve_media_token(token: str, allowed_root: str | Path) -> Path:
    """解析媒体 token，并确保路径仍在 allowed_root 下。"""
    try:
        padded = token + ("=" * (-len(token) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("Invalid media token") from exc

    resolved = Path(decoded).resolve()
    root = Path(allowed_root).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Media path is outside the allowed root") from exc
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def _standardize_event(
    metadata: dict,
    sqlite_event: Optional[dict] = None,
) -> dict:
    sqlite_event = sqlite_event or {}
    event_id = str(metadata.get("event_id") or sqlite_event.get("event_id") or "")
    candidate = _dict_or_empty(sqlite_event.get("candidate")) or _dict_or_empty(
        metadata.get("candidate")
    )
    verification = _dict_or_empty(sqlite_event.get("verification")) or _dict_or_empty(
        metadata.get("verification")
    )
    raw_status = str(
        sqlite_event.get("status")
        or metadata.get("status")
        or metadata.get("category")
        or "candidates"
    )
    display_status = _display_status(raw_status, metadata.get("category"))
    clip_path = str(sqlite_event.get("clip_path") or metadata.get("clip_path") or "")
    yolo_score = _safe_float(sqlite_event.get("yolo_score"))
    if yolo_score is None:
        yolo_score = _safe_float(candidate.get("score"))
    privacy_status = str(
        sqlite_event.get("privacy_status")
        or metadata.get("privacy_status")
        or "raw_unprotected"
    )
    integrity_status = str(
        sqlite_event.get("integrity_status")
        or metadata.get("integrity_status")
        or "not_hashed"
    )
    retention_status = str(
        sqlite_event.get("retention_status")
        or metadata.get("retention_status")
        or "pending_manifest"
    )
    duration_ms = _safe_float(metadata.get("duration_ms"))
    media_url = f"/media/{media_token_for_path(clip_path)}" if clip_path else None

    return {
        "event_id": event_id,
        "camera_id": str(sqlite_event.get("camera_id") or metadata.get("camera_id") or ""),
        "area_label": "模拟区域",
        "source_uri": str(
            sqlite_event.get("source_uri") or metadata.get("source_uri") or ""
        ),
        "clip_path": clip_path,
        "metadata_path": str(
            sqlite_event.get("metadata_path") or metadata.get("metadata_path") or ""
        ),
        "media_url": media_url,
        "category": str(metadata.get("category") or ""),
        "raw_status": raw_status,
        "display_status": display_status,
        "status_label": CATEGORY_LABELS.get(display_status, display_status),
        "created_at": str(
            sqlite_event.get("created_at")
            or metadata.get("created_at")
            or metadata.get("saved_at")
            or ""
        ),
        "updated_at": str(sqlite_event.get("updated_at") or metadata.get("updated_at") or ""),
        "duration_ms": duration_ms,
        "duration_seconds": _round(duration_ms / 1000.0) if duration_ms else None,
        "frame_count": metadata.get("frame_count"),
        "fps": metadata.get("fps"),
        "yolo_score": yolo_score,
        "candidate": candidate,
        "candidate_summary": _candidate_summary(candidate),
        "verification": verification,
        "vlm_result": verification.get("result"),
        "vlm_confidence": _safe_float(verification.get("confidence")),
        "vlm_reason": verification.get("reason") or verification.get("failure_reason"),
        "visible_evidence": verification.get("visible_evidence") or [],
        "privacy_status": privacy_status,
        "integrity_status": integrity_status,
        "retention_status": retention_status,
        "privacy_label": _status_label("privacy_status", privacy_status),
        "integrity_label": _status_label("integrity_status", integrity_status),
        "retention_label": _status_label("retention_status", retention_status),
    }


def _display_status(raw_status: str, fallback_category: Any = None) -> str:
    status = str(raw_status or fallback_category or "").strip()
    if status in {"confirmed_fall", "need_human_review", "rejected", "candidates"}:
        return status
    if status in {
        event_state.YOLO_CANDIDATE,
        event_state.VLM_PENDING,
        event_state.VLM_PROCESSING,
        event_state.VLM_FAILED,
    }:
        return "candidates"
    if status in {
        event_state.PRIVACY_PENDING,
        event_state.INTEGRITY_PENDING,
        event_state.RETENTION_PENDING,
        event_state.ARCHIVED,
    }:
        return "confirmed_fall"
    return str(fallback_category or "candidates")


def _camera_card(camera_id: str, camera_events: list[dict]) -> dict:
    active_statuses = [
        event["display_status"]
        for event in camera_events
        if event["display_status"] in RISK_PRIORITY
        and event["display_status"] != "rejected"
    ]
    risk_status = min(active_statuses or ["normal"], key=RISK_PRIORITY.get)
    visible_alerts = [
        event for event in camera_events if event["display_status"] in DISPLAY_ALERT_STATUSES
    ]
    latest = visible_alerts[0] if visible_alerts else None
    return {
        "camera_id": camera_id,
        "area_label": "模拟区域",
        "online_status": "模拟在线",
        "risk_status": risk_status,
        "risk_label": CATEGORY_LABELS.get(risk_status, risk_status),
        "last_alert_time": latest.get("created_at") if latest else None,
        "last_alert_type": latest.get("status_label") if latest else None,
        "pending_review_count": sum(
            1 for event in camera_events if event["display_status"] in REVIEW_STATUSES
        ),
        "confirmed_event_count": sum(
            1
            for event in camera_events
            if event["display_status"] == "confirmed_fall"
        ),
        "latest_alert": latest,
        "placeholder_text": CAMERA_PLACEHOLDER_TEXT,
    }


def _load_sqlite_events(queue_db_path: str | Path | None) -> dict[str, dict]:
    if queue_db_path is None:
        return {}
    db_path = Path(queue_db_path)
    if not db_path.exists():
        return {}
    try:
        with sqlite3.connect(str(db_path)) as connection:
            connection.row_factory = sqlite3.Row
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'events'"
            ).fetchone()
            if table is None:
                return {}
            rows = connection.execute("SELECT * FROM events").fetchall()
    except sqlite3.Error:
        return {}

    events: dict[str, dict] = {}
    for row in rows:
        event = dict(row)
        event["candidate"] = _json_object(event.pop("candidate_json", None))
        event["verification"] = _json_object(event.pop("verification_json", None))
        events[str(event.get("event_id") or "")] = event
    return events


def _queue_summary(event_dir: str | Path, queue_db_path: str | Path | None) -> dict:
    return build_metrics_report(event_dir, queue_db_path).get("queue", {})


def _status_explanations(event: dict) -> dict:
    return {
        "privacy_status": _status_label("privacy_status", event.get("privacy_status")),
        "integrity_status": _status_label(
            "integrity_status",
            event.get("integrity_status"),
        ),
        "retention_status": _status_label(
            "retention_status",
            event.get("retention_status"),
        ),
    }


def _status_label(kind: str, value: Any) -> str:
    value = str(value or "")
    return STATUS_EXPLANATIONS.get(kind, {}).get(value, value or "未记录")


def _filter_comment_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _filter_comment_fields(item)
            for key, item in value.items()
            if key not in COMMENT_KEYS and not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_filter_comment_fields(item) for item in value]
    return value


def _sort_events(events: Iterable[dict]) -> list[dict]:
    return sorted(
        events,
        key=lambda event: (
            str(event.get("created_at") or ""),
            str(event.get("event_id") or ""),
        ),
        reverse=True,
    )


def _count_status(events: Iterable[dict], status: str) -> int:
    return sum(1 for event in events if event.get("display_status") == status)


def _summarize_frontend_yolo(events: list[dict]) -> dict:
    scores = [
        event["yolo_score"]
        for event in events
        if _safe_float(event.get("yolo_score")) is not None
    ]
    return {
        "candidates": len(scores),
        "average_score": _round(sum(scores) / len(scores)) if scores else 0.0,
        "max_score": _round(max(scores)) if scores else 0.0,
    }


def _summarize_frontend_vlm(events: list[dict]) -> dict:
    verified = [
        event
        for event in events
        if isinstance(event.get("verification"), dict) and event["verification"]
    ]
    confidences = [
        event["vlm_confidence"]
        for event in verified
        if _safe_float(event.get("vlm_confidence")) is not None
    ]
    return {
        "verified_events": len(verified),
        "confirmed_fall": sum(
            1 for event in verified if event.get("vlm_result") == "confirmed_fall"
        ),
        "rejected": sum(1 for event in verified if event.get("vlm_result") == "rejected"),
        "need_human_review": sum(
            1
            for event in verified
            if event.get("vlm_result") == "need_human_review"
        ),
        "average_confidence": (
            _round(sum(confidences) / len(confidences)) if confidences else 0.0
        ),
    }


def _json_object(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dict_or_empty(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _candidate_summary(candidate: dict) -> str:
    score = _safe_float(candidate.get("score"))
    timestamp_ms = candidate.get("timestamp_ms")
    parts = []
    if score is not None:
        parts.append(f"YOLO score {score:.2f}")
    if timestamp_ms is not None:
        parts.append(f"timestamp {timestamp_ms} ms")
    return ", ".join(parts)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)
