"""
统一管理摔倒候选事件和 VLM 复核任务的 SQLite 持久化存储，用于保存候选片段信息、任务队列状态、VLM 结果和事件状态，保证后续异步处理可以在进程重启后继续恢复。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from services import event_state


JOB_PENDING = "pending"
JOB_PROCESSING = "processing"
JOB_DONE = "done"
JOB_FAILED = "failed"


class EventRepositoryError(RuntimeError):
    """Raised when event repository operations fail."""


class EventRepository:
    """SQLite repository for fall events and asynchronous VLM jobs."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        if not str(db_path).strip():
            raise ValueError("db_path must be provided")
        self.db_path = Path(db_path)

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
                """
            )
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
                        clip_path,
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

            connection.execute(
                """
                UPDATE events
                SET status = ?,
                    verification_json = ?,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (final_status, _json_dumps(verification), now, event_id),
            )
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
                """
                UPDATE events
                SET status = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (event_status, now, event_id),
            )

        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Return one event by ID, or None when it does not exist."""
        _require_non_empty("event_id", event_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return _event_row_to_dict(row) if row is not None else None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return one VLM job by ID, or None when it does not exist."""
        _require_non_empty("job_id", job_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vlm_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
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

    def _ensure_event_status(self, status: str) -> None:
        if status not in event_state.ALL_EVENT_STATUSES:
            raise ValueError(f"Unknown event status: {status}")


def _event_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    verification_json = row["verification_json"]
    return {
        "event_id": row["event_id"],
        "camera_id": row["camera_id"],
        "source_uri": row["source_uri"],
        "clip_path": row["clip_path"],
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
]
