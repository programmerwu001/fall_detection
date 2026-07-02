"""Read-only data helpers for the local fall monitoring frontend."""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from services.alert_policy import (
    ALERT_HANDLED,
    ALERT_NONE,
    ALERT_PENDING,
    HIGH_RISK,
    LOW_RISK,
    NO_ALARM,
    map_vlm_decision,
)
from services import event_state
from services.metrics_report import build_metrics_report, load_event_metadata

try:
    from config import PRIVACY_PREVIEW_DIR
except Exception:  # pragma: no cover - standalone import fallback
    PRIVACY_PREVIEW_DIR = Path("data") / "privacy_previews"


SIMULATION_NOTE = (
    "当前原型使用本地视频模拟摄像头输入，因此本页暂不展示持续实时画面；"
    "接入 RTSP 或真实摄像头后，可替换为实时视频流。"
)
CAMERA_PLACEHOLDER_TEXT = "当前为模拟摄像头\n暂无实时画面\n最近告警可在回放中查看"

COMMENT_KEYS = {"_说明", "参数说明"}
LEGACY_READ_ONLY = "legacy_read_only"

RISK_PRIORITY = {
    HIGH_RISK: 0,
    LOW_RISK: 1,
    ALERT_HANDLED: 2,
    "normal": 3,
}

CATEGORY_LABELS = {
    HIGH_RISK: "高风险摔倒告警",
    LOW_RISK: "低风险摔倒告警",
    ALERT_HANDLED: "已处理",
    NO_ALARM: "无护工告警",
    "rejected": "无护工告警",
    "normal": "正常",
    "pending_detection": "检测处理中",
}

PUBLIC_EVENT_FIELDS = (
    "event_id",
    "camera_id",
    "area_label",
    "display_status",
    "status_label",
    "risk_level",
    "risk_label",
    "alert_status",
    "can_handle",
    "handled_by",
    "handled_at",
    "last_notified_at",
    "next_remind_at",
    "reminder_count",
    "decision_source",
    "system_degraded",
    "vlm_label",
    "privacy_preview_status",
    "privacy_preview_url",
    "created_at",
    "updated_at",
    "duration_ms",
    "duration_seconds",
    "frame_count",
    "fps",
    "yolo_score",
    "candidate",
    "candidate_summary",
    "verification",
    "vlm_confidence",
    "vlm_reason",
    "visible_evidence",
    "privacy_status",
    "integrity_status",
    "retention_status",
    "privacy_label",
    "integrity_label",
    "retention_label",
)

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
            1 for camera in cameras if camera["risk_status"] in {HIGH_RISK, LOW_RISK}
        ),
        "high_risk": _count_risk(events, HIGH_RISK),
        "low_risk": _count_risk(events, LOW_RISK),
        "handled": _count_alert_status(events, ALERT_HANDLED),
        "no_alarm": _count_risk(events, NO_ALARM),
        "pending_detection": _count_risk(events, "pending_detection"),
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
    """返回告警中心列表，不包含 no_alarm。"""
    return [
        event
        for event in list_events(event_dir, queue_db_path)
        if event["risk_level"] in {HIGH_RISK, LOW_RISK}
    ]


def fall_events(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回高风险摔倒告警。"""
    return [
        event
        for event in list_events(event_dir, queue_db_path)
        if event["risk_level"] == HIGH_RISK
    ]


def review_alerts(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回低风险摔倒告警。"""
    return [
        event
        for event in list_events(event_dir, queue_db_path)
        if event["risk_level"] == LOW_RISK
    ]


def showcase_cases(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> list[dict]:
    """返回可点击展示案例，不包含 no_alarm。"""
    return alerts(event_dir, queue_db_path)


def evaluation_summary(
    event_dir: str | Path,
    queue_db_path: str | Path | None = None,
) -> dict:
    """返回案例回放与评估模块使用的指标摘要，包含 no_alarm 数量。"""
    events = list_events(event_dir, queue_db_path)
    report = build_metrics_report(event_dir, queue_db_path)
    return {
        "displayed_cases": sum(
            1 for event in events if event["risk_level"] in {HIGH_RISK, LOW_RISK}
        ),
        "high_risk": _count_risk(events, HIGH_RISK),
        "low_risk": _count_risk(events, LOW_RISK),
        "handled": _count_alert_status(events, ALERT_HANDLED),
        "pending_detection": _count_risk(events, "pending_detection"),
        "no_alarm": _count_risk(events, NO_ALARM),
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
                "candidate": _public_nested(event.get("candidate") or {}),
                "verification": _public_nested(event.get("verification") or {}),
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


def media_token_for_path(path: str | Path, allowed_root: str | Path | None = None) -> str:
    """Encode a media URL token without exposing local paths when a root is given."""
    resolved = Path(path).resolve()
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            value = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Media path is outside the allowed root") from exc
    else:
        value = str(resolved)
    encoded = base64.urlsafe_b64encode(value.encode("utf-8"))
    return encoded.decode("ascii")


def resolve_media_token(token: str, allowed_root: str | Path) -> Path:
    """解析媒体 token，并确保路径仍在 allowed_root 下。"""
    try:
        padded = token + ("=" * (-len(token) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("Invalid media token") from exc

    root = Path(allowed_root).resolve()
    decoded_path = Path(decoded)
    resolved = (
        decoded_path.resolve()
        if decoded_path.is_absolute()
        else (root / decoded_path).resolve()
    )
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
    sqlite_backed = bool(sqlite_event.get("_sqlite_backed"))
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
    risk_level = _risk_level(sqlite_event, metadata, raw_status)
    alert_status = _alert_status(sqlite_event, metadata, risk_level, sqlite_backed)
    display_status = _display_status(
        raw_status=raw_status,
        fallback_category=metadata.get("category"),
        risk_level=risk_level,
        alert_status=alert_status,
    )
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

    privacy_preview_status = str(
        sqlite_event.get("privacy_preview_status")
        or metadata.get("privacy_preview_status")
        or "not_generated"
    )
    privacy_preview_url = _privacy_preview_url(
        privacy_preview_status=privacy_preview_status,
        sqlite_event=sqlite_event,
        metadata=metadata,
    )

    event = {
        "event_id": event_id,
        "camera_id": str(sqlite_event.get("camera_id") or metadata.get("camera_id") or ""),
        "area_label": "模拟区域",
        "category": str(metadata.get("category") or ""),
        "raw_status": raw_status,
        "display_status": display_status,
        "status_label": CATEGORY_LABELS.get(display_status, display_status),
        "risk_level": risk_level,
        "risk_label": CATEGORY_LABELS.get(risk_level, risk_level),
        "alert_status": alert_status,
        "handled_by": sqlite_event.get("handled_by") or metadata.get("handled_by"),
        "handled_at": sqlite_event.get("handled_at") or metadata.get("handled_at"),
        "last_notified_at": sqlite_event.get("last_notified_at"),
        "next_remind_at": sqlite_event.get("next_remind_at"),
        "reminder_count": _safe_int(sqlite_event.get("reminder_count")) or 0,
        "decision_source": sqlite_event.get("decision_source")
        or metadata.get("decision_source")
        or _decision_source_for_legacy(risk_level),
        "system_degraded": bool(
            sqlite_event.get("system_degraded") or metadata.get("system_degraded") or False
        ),
        "can_handle": bool(
            sqlite_backed
            and alert_status == ALERT_PENDING
            and risk_level in {HIGH_RISK, LOW_RISK}
        ),
        "vlm_status": sqlite_event.get("vlm_status") or _vlm_status(verification, raw_status),
        "vlm_label": CATEGORY_LABELS.get(risk_level, CATEGORY_LABELS.get(display_status, "未分级")),
        "privacy_preview_status": privacy_preview_status,
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
        "candidate": _public_candidate(candidate),
        "candidate_summary": _candidate_summary(candidate),
        "verification": _public_verification(verification),
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
    if privacy_preview_url:
        event["privacy_preview_url"] = privacy_preview_url
    return _public_event(event)


def _display_status(
    raw_status: str,
    fallback_category: Any = None,
    risk_level: str | None = None,
    alert_status: str | None = None,
) -> str:
    if alert_status == ALERT_HANDLED:
        return ALERT_HANDLED
    if risk_level in {HIGH_RISK, LOW_RISK, NO_ALARM}:
        return risk_level
    status = str(raw_status or fallback_category or "").strip()
    if status == "confirmed_fall":
        return HIGH_RISK
    if status in {"need_human_review", "uncertain"}:
        return LOW_RISK
    if status == "rejected":
        return NO_ALARM
    if status == "candidates":
        return "pending_detection"
    if status in {
        event_state.YOLO_CANDIDATE,
        event_state.VLM_PENDING,
        event_state.VLM_PROCESSING,
        event_state.VLM_FAILED,
    }:
        return "pending_detection"
    if status in {
        event_state.PRIVACY_PENDING,
        event_state.INTEGRITY_PENDING,
        event_state.RETENTION_PENDING,
        event_state.ARCHIVED,
    }:
        return HIGH_RISK
    return "pending_detection"


def _camera_card(camera_id: str, camera_events: list[dict]) -> dict:
    active_statuses = [
        event["display_status"]
        for event in camera_events
        if event["display_status"] in RISK_PRIORITY
        and event.get("alert_status") == ALERT_PENDING
    ]
    risk_status = min(active_statuses or ["normal"], key=RISK_PRIORITY.get)
    visible_alerts = [
        event for event in camera_events if event["risk_level"] in {HIGH_RISK, LOW_RISK}
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
            1
            for event in camera_events
            if event["risk_level"] == LOW_RISK and event.get("alert_status") == ALERT_PENDING
        ),
        "confirmed_event_count": sum(
            1
            for event in camera_events
            if event["risk_level"] == HIGH_RISK
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
            has_preview_table = (
                connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name = 'privacy_previews'
                    """
                ).fetchone()
                is not None
            )
            if has_preview_table:
                rows = connection.execute(
                    """
                    SELECT events.*,
                           privacy_previews.status AS privacy_preview_status,
                           privacy_previews.preview_path AS privacy_preview_path,
                           privacy_previews.last_error AS privacy_preview_error
                    FROM events
                    LEFT JOIN privacy_previews
                      ON privacy_previews.event_id = events.event_id
                    """
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM events").fetchall()
    except sqlite3.Error:
        return {}

    events: dict[str, dict] = {}
    for row in rows:
        event = dict(row)
        event["_sqlite_backed"] = True
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


def _count_risk(events: Iterable[dict], risk_level: str) -> int:
    return sum(1 for event in events if event.get("risk_level") == risk_level)


def _count_alert_status(events: Iterable[dict], alert_status: str) -> int:
    return sum(1 for event in events if event.get("alert_status") == alert_status)


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
        "high_risk": sum(1 for event in verified if event.get("risk_level") == HIGH_RISK),
        "low_risk": sum(1 for event in verified if event.get("risk_level") == LOW_RISK),
        "no_alarm": sum(1 for event in verified if event.get("risk_level") == NO_ALARM),
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


def _risk_level(sqlite_event: dict, metadata: dict, raw_status: str) -> str:
    stored = sqlite_event.get("risk_level") or metadata.get("risk_level")
    if stored:
        return str(stored)
    verification = _dict_or_empty(sqlite_event.get("verification")) or _dict_or_empty(
        metadata.get("verification")
    )
    result = verification.get("result") or raw_status or metadata.get("category")
    if str(result or "") in {
        "candidates",
        event_state.YOLO_CANDIDATE,
        event_state.VLM_PENDING,
        event_state.VLM_PROCESSING,
    }:
        return "pending_detection"
    return map_vlm_decision(str(result or "")).risk_level


def _alert_status(
    sqlite_event: dict,
    metadata: dict,
    risk_level: str,
    sqlite_backed: bool,
) -> str:
    stored = sqlite_event.get("alert_status") if sqlite_backed else None
    if stored and stored != ALERT_NONE:
        return str(stored)
    if risk_level in {HIGH_RISK, LOW_RISK}:
        return ALERT_PENDING if sqlite_backed else LEGACY_READ_ONLY
    if stored:
        return str(stored)
    return ALERT_NONE


def _decision_source_for_legacy(risk_level: str) -> str | None:
    return "vlm" if risk_level in {HIGH_RISK, LOW_RISK, NO_ALARM} else None


def _vlm_status(verification: dict, raw_status: str) -> str | None:
    result = verification.get("result")
    if result:
        return str(result)
    status = str(raw_status or "")
    if status.startswith("vlm_"):
        return status.removeprefix("vlm_")
    return None


def _privacy_preview_url(
    privacy_preview_status: str,
    sqlite_event: dict,
    metadata: dict,
) -> str | None:
    if privacy_preview_status != "ready":
        return None
    preview_path = (
        sqlite_event.get("privacy_preview_path")
        or metadata.get("privacy_preview_path")
    )
    if not preview_path:
        return None
    path = Path(str(preview_path))
    root = Path(PRIVACY_PREVIEW_DIR)
    try:
        token = media_token_for_path(path, allowed_root=root)
    except ValueError:
        return None
    return f"/media/{token}"


def _public_event(event: dict) -> dict:
    return {key: event.get(key) for key in PUBLIC_EVENT_FIELDS if key in event}


def _public_candidate(candidate: dict) -> dict:
    return _public_nested(candidate) if isinstance(candidate, dict) else {}


def _public_verification(verification: dict) -> dict:
    if not isinstance(verification, dict) or not verification:
        return {}
    allowed = {
        "confidence",
        "reason",
        "failure_reason",
        "visible_evidence",
        "model_id",
        "timestamp_ms",
        "is_confirmed",
    }
    return {
        key: _public_nested(verification.get(key))
        for key in allowed
        if key in verification
    }


def _public_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_nested(item)
            for key, item in value.items()
            if key
            not in {
                "clip_path",
                "debug_clip_path",
                "debug_metadata_path",
                "media_url",
                "metadata_path",
                "raw_response",
                "source_uri",
                "privacy_preview_path",
            }
        }
    if isinstance(value, list):
        return [_public_nested(item) for item in value]
    return value


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)
