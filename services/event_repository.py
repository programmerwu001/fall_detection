"""
统一管理摔倒候选事件和 VLM 复核任务的 SQLite 持久化存储，用于保存候选片段信息、任务队列状态、VLM 结果和事件状态，保证后续异步处理可以在进程重启后继续恢复。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from services.alert_policy import (
    ALERT_HANDLED,
    ALERT_NONE,
    ALERT_PENDING,
    HIGH_RISK,
    LOW_RISK,
    map_vlm_decision,
    reminder_interval_seconds,
)
from services import event_state


JOB_PENDING = "pending"
JOB_PROCESSING = "processing"
JOB_DONE = "done"
JOB_FAILED = "failed"

PRIVACY_PREVIEW_NOT_GENERATED = "not_generated"
PRIVACY_PREVIEW_PENDING = "pending"
PRIVACY_PREVIEW_PROCESSING = "processing"
PRIVACY_PREVIEW_READY = "ready"
PRIVACY_PREVIEW_FAILED = "failed"


class EventRepositoryError(RuntimeError):
    """Raised when event repository operations fail."""


class EventRepository:
    """SQLite repository for fall events and asynchronous VLM jobs."""

    def __init__(
        self,
        db_path: Union[str, Path],
        high_risk_repeat_seconds: int = 20,
        low_risk_repeat_seconds: int = 60,
    ) -> None:
        if not str(db_path).strip():
            raise ValueError("db_path must be provided")
        self.db_path = Path(db_path)
        self.high_risk_repeat_seconds = int(high_risk_repeat_seconds)
        self.low_risk_repeat_seconds = int(low_risk_repeat_seconds)

    def initialize(self) -> "EventRepository":
        """Create database parent directory and required tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    clip_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    yolo_score REAL,
                    candidate_json TEXT NOT NULL,
                    verification_json TEXT,
                    privacy_status TEXT NOT NULL,
                    integrity_status TEXT NOT NULL,
                    retention_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS private_event_clips (
                    event_id TEXT PRIMARY KEY,
                    clip_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_status_created_at
                ON events(status, created_at);

                CREATE INDEX IF NOT EXISTS idx_events_camera_created_at
                ON events(camera_id, created_at);

                CREATE TABLE IF NOT EXISTS vlm_jobs (
                    job_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    locked_by TEXT,
                    locked_until TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_vlm_jobs_status_priority_created_at
                ON vlm_jobs(status, priority, created_at);

                CREATE TABLE IF NOT EXISTS privacy_previews (
                    event_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    preview_path TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE TABLE IF NOT EXISTS privacy_preview_jobs (
                    job_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    locked_by TEXT,
                    locked_until TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_privacy_preview_jobs_status_priority_created_at
                ON privacy_preview_jobs(status, priority, created_at);
                """
            )
            self._ensure_event_columns(connection)
            self._migrate_private_clip_paths(connection)
            self._backfill_alert_decision_columns(connection)
        return self

    def create_candidate_event(
        self,
        event_id: str,
        camera_id: str,
        source_uri: str,
        clip_path: str,
        metadata_path: str,
        candidate: Dict[str, Any],
        yolo_score: Optional[float] = None,
        status: str = event_state.VLM_PENDING,
        privacy_status: str = "raw_unprotected",
        integrity_status: str = "not_hashed",
        retention_status: str = "pending_manifest",
    ) -> Dict[str, Any]:
        """Persist one YOLO candidate event and return the stored event."""
        _require_non_empty("event_id", event_id)
        _require_non_empty("camera_id", camera_id)
        _require_non_empty("source_uri", source_uri)
        _require_non_empty("clip_path", clip_path)
        _require_non_empty("metadata_path", metadata_path)
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be a dictionary")
        self._ensure_event_status(status)

        now = _now_iso()
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO events (
                        event_id,
                        camera_id,
                        source_uri,
                        clip_path,
                        metadata_path,
                        status,
                        yolo_score,
                        candidate_json,
                        verification_json,
                        privacy_status,
                        integrity_status,
                        retention_status,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        camera_id,
                        source_uri,
                        "",
                        metadata_path,
                        status,
                        yolo_score,
                        _json_dumps(candidate),
                        privacy_status,
                        integrity_status,
                        retention_status,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO private_event_clips (
                        event_id,
                        clip_path,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (event_id, clip_path, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise EventRepositoryError(
                    f"Event already exists or references are invalid: {event_id}"
                ) from exc

        event = self.get_event(event_id)
        assert event is not None
        return event

    def enqueue_vlm_job(
        self,
        event_id: str,
        job_id: Optional[str] = None,
        priority: int = 100,
    ) -> Dict[str, Any]:
        """Create a pending VLM job for an existing candidate event."""
        _require_non_empty("event_id", event_id)
        job_id = job_id or f"vlm_{event_id}"
        _require_non_empty("job_id", job_id)
        now = _now_iso()

        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO vlm_jobs (
                        job_id,
                        event_id,
                        status,
                        priority,
                        attempts,
                        locked_by,
                        locked_until,
                        last_error,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?)
                    """,
                    (job_id, event_id, JOB_PENDING, int(priority), now, now),
                )
                connection.execute(
                    """
                    UPDATE events
                    SET status = ?, updated_at = ?
                    WHERE event_id = ?
                    """,
                    (event_state.VLM_PENDING, now, event_id),
                )
            except sqlite3.IntegrityError as exc:
                raise EventRepositoryError(
                    f"Cannot enqueue VLM job for event: {event_id}"
                ) from exc

        job = self.get_job(job_id)
        assert job is not None
        return job

    def lease_vlm_job(
        self,
        worker_id: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Lease one pending or expired VLM job and mark its event as processing.

        The update is done inside an immediate transaction so concurrent workers
        cannot lease the same job at the same time.
        """
        _require_non_empty("worker_id", worker_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")

        now_dt = now or datetime.now()
        now_iso = _datetime_iso(now_dt)
        locked_until = _datetime_iso(now_dt + timedelta(seconds=lease_seconds))

        with self._connect() as connection:
            connection.isolation_level = None
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT *
                    FROM vlm_jobs
                    WHERE status = ?
                       OR (status = ? AND locked_until IS NOT NULL AND locked_until <= ?)
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                    """,
                    (JOB_PENDING, JOB_PROCESSING, now_iso),
                ).fetchone()

                if row is None:
                    connection.execute("COMMIT")
                    return None

                job_id = str(row["job_id"])
                event_id = str(row["event_id"])
                connection.execute(
                    """
                    UPDATE vlm_jobs
                    SET status = ?,
                        attempts = attempts + 1,
                        locked_by = ?,
                        locked_until = ?,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (JOB_PROCESSING, worker_id, locked_until, now_iso, job_id),
                )
                connection.execute(
                    """
                    UPDATE events
                    SET status = ?, updated_at = ?
                    WHERE event_id = ?
                    """,
                    (event_state.VLM_PROCESSING, now_iso, event_id),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        return self.get_job(job_id)

    def complete_vlm_job(
        self,
        job_id: str,
        verification: Dict[str, Any],
        final_status: str,
    ) -> Dict[str, Any]:
        """Mark a VLM job done and write the VLM result to its event."""
        _require_non_empty("job_id", job_id)
        if not isinstance(verification, dict):
            raise ValueError("verification must be a dictionary")
        self._ensure_event_status(final_status)
        now = _now_iso()

        with self._connect() as connection:
            row = self._get_job_row(connection, job_id)
            if row is None:
                raise EventRepositoryError(f"VLM job does not exist: {job_id}")
            event_id = str(row["event_id"])
            event_row = connection.execute(
                "SELECT alert_status FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            is_handled = (
                event_row is not None
                and _row_value(event_row, "alert_status") == ALERT_HANDLED
            )

            if is_handled:
                connection.execute(
                    """
                    UPDATE events
                    SET verification_json = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (_json_dumps(verification), now, event_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE events
                    SET status = ?,
                        verification_json = ?,
                        risk_level = ?,
                        alert_status = ?,
                        last_notified_at = NULL,
                        next_remind_at = ?,
                        reminder_count = 0,
                        decision_source = ?,
                        system_degraded = ?,
                        vlm_status = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    _event_decision_update_values(
                        final_status=final_status,
                        verification=verification,
                        now=now,
                        event_id=event_id,
                        high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                        low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                    ),
                )
                if _should_enqueue_privacy_preview(
                    final_status=final_status,
                    verification=verification,
                    high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                    low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                ):
                    self._enqueue_privacy_preview_job(connection, event_id, now)
            connection.execute(
                """
                UPDATE vlm_jobs
                SET status = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_DONE, now, job_id),
            )

        job = self.get_job(job_id)
        assert job is not None
        return job

    def record_event_decision(
        self,
        event_id: str,
        verification: Dict[str, Any],
        final_status: str,
    ) -> Dict[str, Any]:
        """Write a final event decision when no VLM queue job exists."""
        _require_non_empty("event_id", event_id)
        if not isinstance(verification, dict):
            raise ValueError("verification must be a dictionary")
        self._ensure_event_status(final_status)
        now = _now_iso()

        with self._connect() as connection:
            event_row = connection.execute(
                "SELECT alert_status FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if event_row is None:
                raise EventRepositoryError(f"Event does not exist: {event_id}")
            is_handled = _row_value(event_row, "alert_status") == ALERT_HANDLED
            if is_handled:
                connection.execute(
                    """
                    UPDATE events
                    SET verification_json = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (_json_dumps(verification), now, event_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE events
                    SET status = ?,
                        verification_json = ?,
                        risk_level = ?,
                        alert_status = ?,
                        last_notified_at = NULL,
                        next_remind_at = ?,
                        reminder_count = 0,
                        decision_source = ?,
                        system_degraded = ?,
                        vlm_status = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    _event_decision_update_values(
                        final_status=final_status,
                        verification=verification,
                        now=now,
                        event_id=event_id,
                        high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                        low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                    ),
                )
                if _should_enqueue_privacy_preview(
                    final_status=final_status,
                    verification=verification,
                    high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                    low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                ):
                    self._enqueue_privacy_preview_job(connection, event_id, now)

        event = self.get_event(event_id)
        assert event is not None
        return event

    def mark_vlm_job_degraded(
        self,
        job_id: str,
        reason: str,
        failure_status: str = "failed",
    ) -> Dict[str, Any]:
        """Finish a VLM job as a YOLO fallback low-risk alert immediately."""
        _require_non_empty("job_id", job_id)
        now = _now_iso()
        failure_status = _vlm_failure_status(failure_status or reason)
        verification = {
            "result": failure_status,
            "confidence": 0.0,
            "reason": str(reason),
            "visible_evidence": [],
            "metadata": {"decision_source": "yolo_fallback"},
        }
        with self._connect() as connection:
            row = self._get_job_row(connection, job_id)
            if row is None:
                raise EventRepositoryError(f"VLM job does not exist: {job_id}")
            event_id = str(row["event_id"])
            event_row = connection.execute(
                "SELECT alert_status FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            is_handled = (
                event_row is not None
                and _row_value(event_row, "alert_status") == ALERT_HANDLED
            )

            connection.execute(
                """
                UPDATE vlm_jobs
                SET status = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_FAILED, str(reason), now, job_id),
            )
            if is_handled:
                connection.execute(
                    """
                    UPDATE events
                    SET verification_json = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (_json_dumps(verification), now, event_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE events
                    SET status = ?,
                        verification_json = ?,
                        risk_level = ?,
                        alert_status = ?,
                        last_notified_at = NULL,
                        next_remind_at = ?,
                        reminder_count = 0,
                        decision_source = ?,
                        system_degraded = ?,
                        vlm_status = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    _event_decision_update_values(
                        final_status=event_state.NEED_HUMAN_REVIEW,
                        verification=verification,
                        now=now,
                        event_id=event_id,
                        high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                        low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                    ),
                )
                self._enqueue_privacy_preview_job(connection, event_id, now)

        job = self.get_job(job_id)
        assert job is not None
        return job

    def fail_vlm_job(
        self,
        job_id: str,
        error: str,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Record a VLM job failure and either retry or send event to review."""
        _require_non_empty("job_id", job_id)
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")
        now = _now_iso()
        error_text = str(error)

        with self._connect() as connection:
            row = self._get_job_row(connection, job_id)
            if row is None:
                raise EventRepositoryError(f"VLM job does not exist: {job_id}")

            event_id = str(row["event_id"])
            attempts = int(row["attempts"])
            should_retry = attempts < max_retries
            job_status = JOB_PENDING if should_retry else JOB_FAILED
            event_status = (
                event_state.VLM_PENDING
                if should_retry
                else event_state.NEED_HUMAN_REVIEW
            )
            failure_status = _vlm_failure_status(error_text)

            connection.execute(
                """
                UPDATE vlm_jobs
                SET status = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (job_status, error_text, now, job_id),
            )
            connection.execute(
                _fail_event_sql(should_retry),
                _fail_event_values(
                    should_retry=should_retry,
                    event_status=event_status,
                    failure_status=failure_status,
                    now=now,
                    event_id=event_id,
                    high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                    low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                ),
            )
            if not should_retry:
                self._enqueue_privacy_preview_job(connection, event_id, now)

        job = self.get_job(job_id)
        assert job is not None
        return job

    def enqueue_privacy_preview_job(
        self,
        event_id: str,
        job_id: Optional[str] = None,
        priority: int = 100,
    ) -> Dict[str, Any]:
        """Queue asynchronous silhouette preview generation for one alert event."""
        _require_non_empty("event_id", event_id)
        now = _now_iso()
        with self._connect() as connection:
            event_row = connection.execute(
                "SELECT event_id FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if event_row is None:
                raise EventRepositoryError(f"Event does not exist: {event_id}")
            self._enqueue_privacy_preview_job(
                connection=connection,
                event_id=event_id,
                now=now,
                job_id=job_id,
                priority=priority,
            )
        job = self.get_privacy_preview_job(job_id or f"privacy_{event_id}")
        assert job is not None
        return job

    def lease_privacy_preview_job(
        self,
        worker_id: str,
        lease_seconds: int,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Lease one privacy preview job without blocking the alert lifecycle."""
        _require_non_empty("worker_id", worker_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")

        now_dt = now or datetime.now()
        now_iso = _datetime_iso(now_dt)
        locked_until = _datetime_iso(now_dt + timedelta(seconds=lease_seconds))

        with self._connect() as connection:
            connection.isolation_level = None
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT *
                    FROM privacy_preview_jobs
                    WHERE status = ?
                       OR (status = ? AND locked_until IS NOT NULL AND locked_until <= ?)
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                    """,
                    (JOB_PENDING, JOB_PROCESSING, now_iso),
                ).fetchone()

                if row is None:
                    connection.execute("COMMIT")
                    return None

                job_id = str(row["job_id"])
                event_id = str(row["event_id"])
                connection.execute(
                    """
                    UPDATE privacy_preview_jobs
                    SET status = ?,
                        attempts = attempts + 1,
                        locked_by = ?,
                        locked_until = ?,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (JOB_PROCESSING, worker_id, locked_until, now_iso, job_id),
                )
                connection.execute(
                    """
                    UPDATE privacy_previews
                    SET status = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (PRIVACY_PREVIEW_PROCESSING, now_iso, event_id),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

        return self.get_privacy_preview_job(job_id)

    def complete_privacy_preview_job(
        self,
        job_id: str,
        preview_path: str,
    ) -> Dict[str, Any]:
        """Mark a privacy preview job done and record its dedicated preview path."""
        _require_non_empty("job_id", job_id)
        _require_non_empty("preview_path", preview_path)
        now = _now_iso()
        with self._connect() as connection:
            row = self._get_privacy_preview_job_row(connection, job_id)
            if row is None:
                raise EventRepositoryError(f"Privacy preview job does not exist: {job_id}")
            event_id = str(row["event_id"])
            connection.execute(
                """
                UPDATE privacy_previews
                SET status = ?,
                    preview_path = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (PRIVACY_PREVIEW_READY, str(preview_path), now, event_id),
            )
            connection.execute(
                """
                UPDATE privacy_preview_jobs
                SET status = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (JOB_DONE, now, job_id),
            )

        job = self.get_privacy_preview_job(job_id)
        assert job is not None
        return job

    def fail_privacy_preview_job(
        self,
        job_id: str,
        error: str,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """Record preview generation failure without changing alert state."""
        _require_non_empty("job_id", job_id)
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")
        now = _now_iso()
        error_text = str(error)
        with self._connect() as connection:
            row = self._get_privacy_preview_job_row(connection, job_id)
            if row is None:
                raise EventRepositoryError(f"Privacy preview job does not exist: {job_id}")
            event_id = str(row["event_id"])
            attempts = int(row["attempts"])
            should_retry = attempts < max_retries
            job_status = JOB_PENDING if should_retry else JOB_FAILED
            preview_status = (
                PRIVACY_PREVIEW_PENDING if should_retry else PRIVACY_PREVIEW_FAILED
            )
            connection.execute(
                """
                UPDATE privacy_preview_jobs
                SET status = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    last_error = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (job_status, error_text, now, job_id),
            )
            connection.execute(
                """
                UPDATE privacy_previews
                SET status = ?,
                    last_error = ?,
                    preview_path = CASE WHEN ? THEN preview_path ELSE NULL END,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (preview_status, error_text, int(should_retry), now, event_id),
            )

        job = self.get_privacy_preview_job(job_id)
        assert job is not None
        return job

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Return one event by ID, or None when it does not exist."""
        _require_non_empty("event_id", event_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT events.*,
                       private_event_clips.clip_path AS private_clip_path,
                       privacy_previews.status AS privacy_preview_status,
                       privacy_previews.preview_path AS privacy_preview_path,
                       privacy_previews.last_error AS privacy_preview_error
                FROM events
                LEFT JOIN private_event_clips
                  ON private_event_clips.event_id = events.event_id
                LEFT JOIN privacy_previews
                  ON privacy_previews.event_id = events.event_id
                WHERE events.event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return _event_row_to_dict(row) if row is not None else None

    def mark_event_handled(
        self,
        event_id: str,
        handled_by: str,
        handled_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically mark a pending caregiver alert as handled."""
        _require_non_empty("event_id", event_id)
        _require_non_empty("handled_by", handled_by)
        handled_at = handled_at or _now_iso()
        now = _now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE events
                SET alert_status = ?,
                    handled_by = ?,
                    handled_at = ?,
                    next_remind_at = NULL,
                    updated_at = ?
                WHERE event_id = ? AND alert_status = ?
                """,
                (ALERT_HANDLED, handled_by, handled_at, now, event_id, ALERT_PENDING),
            )
        event = self.get_event(event_id)
        if event is None:
            raise EventRepositoryError(f"Event does not exist: {event_id}")
        return event

    def claim_due_reminders(self, now: Optional[Union[str, datetime]] = None) -> list[Dict[str, Any]]:
        """Return due pending alerts and move their next reminder time forward."""
        now_iso = _coerce_iso(now)
        due_ids: list[str] = []
        with self._connect() as connection:
            connection.isolation_level = None
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT event_id, risk_level
                    FROM events
                    WHERE alert_status = ?
                      AND next_remind_at IS NOT NULL
                      AND next_remind_at <= ?
                    ORDER BY next_remind_at ASC, created_at ASC
                    """,
                    (ALERT_PENDING, now_iso),
                ).fetchall()
                for row in rows:
                    event_id = str(row["event_id"])
                    interval = reminder_interval_seconds(
                        row["risk_level"],
                        high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                        low_risk_repeat_seconds=self.low_risk_repeat_seconds,
                    )
                    if interval is None:
                        continue
                    next_remind_at = _datetime_iso(
                        _parse_iso(now_iso) + timedelta(seconds=interval)
                    )
                    cursor = connection.execute(
                        """
                        UPDATE events
                        SET last_notified_at = ?,
                            next_remind_at = ?,
                            reminder_count = reminder_count + 1,
                            updated_at = ?
                        WHERE event_id = ? AND alert_status = ?
                        """,
                        (
                            now_iso,
                            next_remind_at,
                            now_iso,
                            event_id,
                            ALERT_PENDING,
                        ),
                    )
                    if cursor.rowcount:
                        due_ids.append(event_id)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return [event for event_id in due_ids if (event := self.get_event(event_id))]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return one VLM job by ID, or None when it does not exist."""
        _require_non_empty("job_id", job_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vlm_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _job_row_to_dict(row) if row is not None else None

    def get_privacy_preview_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return one privacy preview job by ID, or None when it does not exist."""
        _require_non_empty("job_id", job_id)
        with self._connect() as connection:
            row = self._get_privacy_preview_job_row(connection, job_id)
        return _job_row_to_dict(row) if row is not None else None

    def get_queue_stats(self) -> Dict[str, int]:
        """Return VLM job counts grouped by status."""
        stats = {
            JOB_PENDING: 0,
            JOB_PROCESSING: 0,
            JOB_DONE: 0,
            JOB_FAILED: 0,
        }
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM vlm_jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            stats[str(row["status"])] = int(row["count"])
        return stats

    def get_privacy_preview_queue_stats(self) -> Dict[str, int]:
        """Return privacy preview job counts grouped by status."""
        stats = {
            JOB_PENDING: 0,
            JOB_PROCESSING: 0,
            JOB_DONE: 0,
            JOB_FAILED: 0,
        }
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM privacy_preview_jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            stats[str(row["status"])] = int(row["count"])
        return stats

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _get_job_row(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT * FROM vlm_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()

    def _get_privacy_preview_job_row(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT * FROM privacy_preview_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()

    def _enqueue_privacy_preview_job(
        self,
        connection: sqlite3.Connection,
        event_id: str,
        now: str,
        job_id: Optional[str] = None,
        priority: int = 100,
    ) -> None:
        job_id = job_id or f"privacy_{event_id}"
        preview_row = connection.execute(
            "SELECT status FROM privacy_previews WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if preview_row is None:
            connection.execute(
                """
                INSERT INTO privacy_previews (
                    event_id,
                    status,
                    preview_path,
                    last_error,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, NULL, NULL, ?, ?)
                """,
                (event_id, PRIVACY_PREVIEW_PENDING, now, now),
            )
        elif str(preview_row["status"]) != PRIVACY_PREVIEW_READY:
            connection.execute(
                """
                UPDATE privacy_previews
                SET status = ?,
                    preview_path = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (PRIVACY_PREVIEW_PENDING, now, event_id),
            )

        job_row = connection.execute(
            "SELECT status FROM privacy_preview_jobs WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if job_row is None:
            connection.execute(
                """
                INSERT INTO privacy_preview_jobs (
                    job_id,
                    event_id,
                    status,
                    priority,
                    attempts,
                    locked_by,
                    locked_until,
                    last_error,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?)
                """,
                (job_id, event_id, JOB_PENDING, int(priority), now, now),
            )
        elif str(job_row["status"]) != JOB_DONE:
            connection.execute(
                """
                UPDATE privacy_preview_jobs
                SET status = ?,
                    priority = ?,
                    locked_by = NULL,
                    locked_until = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (JOB_PENDING, int(priority), now, event_id),
            )

    def _ensure_event_status(self, status: str) -> None:
        if status not in event_state.ALL_EVENT_STATUSES:
            raise ValueError(f"Unknown event status: {status}")

    def _ensure_event_columns(self, connection: sqlite3.Connection) -> None:
        existing = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        columns = {
            "risk_level": "TEXT",
            "alert_status": f"TEXT NOT NULL DEFAULT '{ALERT_NONE}'",
            "handled_by": "TEXT",
            "handled_at": "TEXT",
            "last_notified_at": "TEXT",
            "next_remind_at": "TEXT",
            "reminder_count": "INTEGER NOT NULL DEFAULT 0",
            "decision_source": "TEXT",
            "system_degraded": "INTEGER NOT NULL DEFAULT 0",
            "vlm_status": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")

    def _migrate_private_clip_paths(self, connection: sqlite3.Connection) -> None:
        now = _now_iso()
        connection.execute(
            """
            INSERT OR IGNORE INTO private_event_clips (
                event_id,
                clip_path,
                created_at,
                updated_at
            )
            SELECT event_id,
                   clip_path,
                   COALESCE(created_at, ?),
                   COALESCE(updated_at, ?)
            FROM events
            WHERE clip_path IS NOT NULL AND clip_path != ''
            """,
            (now, now),
        )
        connection.execute(
            "UPDATE events SET clip_path = '' WHERE clip_path IS NOT NULL AND clip_path != ''"
        )

    def _backfill_alert_decision_columns(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT event_id,
                   status,
                   verification_json,
                   risk_level,
                   alert_status,
                   next_remind_at
            FROM events
            """
        ).fetchall()
        now = _now_iso()
        for row in rows:
            if _row_value(row, "alert_status") == ALERT_HANDLED:
                continue
            verification = _json_loads_or_empty(_row_value(row, "verification_json"))
            decision_result = _decision_result_for_existing_event(
                status=str(row["status"] or ""),
                verification=verification,
            )
            if not decision_result:
                continue
            decision = map_vlm_decision(
                decision_result,
                high_risk_repeat_seconds=self.high_risk_repeat_seconds,
                low_risk_repeat_seconds=self.low_risk_repeat_seconds,
            )
            current_risk = _row_value(row, "risk_level")
            current_alert = _row_value(row, "alert_status")
            current_next_remind = _row_value(row, "next_remind_at")
            if (
                current_risk
                and current_alert
                and current_alert != ALERT_NONE
                and (decision.alert_status != ALERT_PENDING or current_next_remind)
            ):
                continue
            next_remind_at = (
                current_next_remind
                if current_next_remind
                else now if decision.alert_status == ALERT_PENDING else None
            )
            connection.execute(
                """
                UPDATE events
                SET risk_level = ?,
                    alert_status = ?,
                    next_remind_at = ?,
                    decision_source = COALESCE(decision_source, ?),
                    system_degraded = ?,
                    vlm_status = COALESCE(vlm_status, ?),
                    updated_at = COALESCE(updated_at, ?)
                WHERE event_id = ?
                """,
                (
                    decision.risk_level,
                    decision.alert_status,
                    next_remind_at,
                    decision.decision_source,
                    int(decision.system_degraded),
                    decision.vlm_status,
                    now,
                    row["event_id"],
                ),
            )


def _event_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    verification_json = row["verification_json"]
    private_clip_path = _row_value(row, "private_clip_path") or row["clip_path"]
    return {
        "event_id": row["event_id"],
        "camera_id": row["camera_id"],
        "source_uri": row["source_uri"],
        "clip_path": private_clip_path,
        "metadata_path": row["metadata_path"],
        "status": row["status"],
        "yolo_score": row["yolo_score"],
        "candidate": _json_loads(row["candidate_json"]),
        "verification": (
            _json_loads(verification_json) if verification_json is not None else None
        ),
        "privacy_status": row["privacy_status"],
        "integrity_status": row["integrity_status"],
        "retention_status": row["retention_status"],
        "risk_level": _row_value(row, "risk_level"),
        "alert_status": _row_value(row, "alert_status") or ALERT_NONE,
        "handled_by": _row_value(row, "handled_by"),
        "handled_at": _row_value(row, "handled_at"),
        "last_notified_at": _row_value(row, "last_notified_at"),
        "next_remind_at": _row_value(row, "next_remind_at"),
        "reminder_count": int(_row_value(row, "reminder_count") or 0),
        "decision_source": _row_value(row, "decision_source"),
        "system_degraded": bool(_row_value(row, "system_degraded") or 0),
        "vlm_status": _row_value(row, "vlm_status"),
        "privacy_preview_status": _row_value(row, "privacy_preview_status")
        or PRIVACY_PREVIEW_NOT_GENERATED,
        "privacy_preview_path": _row_value(row, "privacy_preview_path"),
        "privacy_preview_error": _row_value(row, "privacy_preview_error"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _job_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "event_id": row["event_id"],
        "status": row["status"],
        "priority": row["priority"],
        "attempts": row["attempts"],
        "locked_by": row["locked_by"],
        "locked_until": row["locked_until"],
        "last_error": row["last_error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _require_non_empty(name: str, value: str) -> None:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be provided")


def _json_dumps(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)


def _json_loads(value: str) -> Dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise EventRepositoryError("Stored JSON value must be an object")
    return loaded


def _json_loads_or_empty(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    try:
        loaded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _decision_result_for_existing_event(
    status: str,
    verification: Dict[str, Any],
) -> Optional[str]:
    result = verification.get("result")
    if result:
        return str(result)
    if status in {
        event_state.CONFIRMED_FALL,
        event_state.NEED_HUMAN_REVIEW,
        event_state.REJECTED,
        event_state.VLM_FAILED,
    }:
        return status
    return None


def _event_decision_update_values(
    final_status: str,
    verification: Dict[str, Any],
    now: str,
    event_id: str,
    high_risk_repeat_seconds: int = 20,
    low_risk_repeat_seconds: int = 60,
) -> tuple:
    decision = map_vlm_decision(
        str(verification.get("result") or final_status),
        high_risk_repeat_seconds=high_risk_repeat_seconds,
        low_risk_repeat_seconds=low_risk_repeat_seconds,
    )
    next_remind_at = now if decision.alert_status == ALERT_PENDING else None
    return (
        final_status,
        _json_dumps(verification),
        decision.risk_level,
        decision.alert_status,
        next_remind_at,
        decision.decision_source,
        int(decision.system_degraded),
        decision.vlm_status,
        now,
        event_id,
    )


def _should_enqueue_privacy_preview(
    final_status: str,
    verification: Dict[str, Any],
    high_risk_repeat_seconds: int = 20,
    low_risk_repeat_seconds: int = 60,
) -> bool:
    decision = map_vlm_decision(
        str(verification.get("result") or final_status),
        high_risk_repeat_seconds=high_risk_repeat_seconds,
        low_risk_repeat_seconds=low_risk_repeat_seconds,
    )
    return (
        decision.alert_status == ALERT_PENDING
        and decision.risk_level in {HIGH_RISK, LOW_RISK}
    )


def _fail_event_sql(should_retry: bool) -> str:
    if should_retry:
        return """
            UPDATE events
            SET status = ?, updated_at = ?
            WHERE event_id = ?
        """
    return """
        UPDATE events
        SET status = ?,
            risk_level = ?,
            alert_status = ?,
            last_notified_at = NULL,
            next_remind_at = ?,
            reminder_count = 0,
            decision_source = ?,
            system_degraded = ?,
            vlm_status = ?,
            updated_at = ?
        WHERE event_id = ?
    """


def _fail_event_values(
    should_retry: bool,
    event_status: str,
    failure_status: str,
    now: str,
    event_id: str,
    high_risk_repeat_seconds: int = 20,
    low_risk_repeat_seconds: int = 60,
) -> tuple:
    if should_retry:
        return event_status, now, event_id
    decision = map_vlm_decision(
        failure_status,
        high_risk_repeat_seconds=high_risk_repeat_seconds,
        low_risk_repeat_seconds=low_risk_repeat_seconds,
    )
    return (
        event_status,
        decision.risk_level,
        decision.alert_status,
        now if decision.alert_status == ALERT_PENDING else None,
        decision.decision_source,
        int(decision.system_degraded),
        decision.vlm_status,
        now,
        event_id,
    )


def _vlm_failure_status(error_text: str) -> str:
    return "timeout" if "timeout" in str(error_text).lower() else "failed"


def _row_value(row: sqlite3.Row, key: str) -> Any:
    return row[key] if key in row.keys() else None


def _coerce_iso(value: Optional[Union[str, datetime]]) -> str:
    if value is None:
        return _now_iso()
    if isinstance(value, datetime):
        return _datetime_iso(value)
    return str(value)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _now_iso() -> str:
    return _datetime_iso(datetime.now())


def _datetime_iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


__all__ = [
    "EventRepository",
    "EventRepositoryError",
    "JOB_PENDING",
    "JOB_PROCESSING",
    "JOB_DONE",
    "JOB_FAILED",
    "PRIVACY_PREVIEW_NOT_GENERATED",
    "PRIVACY_PREVIEW_PENDING",
    "PRIVACY_PREVIEW_PROCESSING",
    "PRIVACY_PREVIEW_READY",
    "PRIVACY_PREVIEW_FAILED",
]
