"""Asynchronous worker for caregiver-safe silhouette privacy previews."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import (
    DB_PATH,
    DEFAULT_PRIVACY_PREVIEW_MODEL,
    PRIVACY_PREVIEW_DIR,
)
from services.event_repository import EventRepository
from services.privacy_preview import PrivacyPreviewGenerator, YoloPersonDetector


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "detection_config.json"
CONFIG_COMMENT_KEYS = {"_注释", "_说明", "_comment", "comment", "参数说明", "使用说明"}


@dataclass
class WorkerStats:
    leased: int = 0
    processed: int = 0
    completed: int = 0
    failed: int = 0
    idle: int = 0
    errors: int = 0

    def add(self, other: "WorkerStats") -> None:
        self.leased += other.leased
        self.processed += other.processed
        self.completed += other.completed
        self.failed += other.failed
        self.idle += other.idle
        self.errors += other.errors


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    logger.info("Privacy preview person model: %s", args.person_model)

    repository = EventRepository(args.queue_db_path).initialize()
    detector = YoloPersonDetector(
        model_path=args.person_model,
        confidence=args.person_confidence,
    )
    logger.info("Loading privacy preview person detector before polling jobs")
    detector.load()
    generator = PrivacyPreviewGenerator(
        preview_root=args.preview_dir,
        detector=detector,
        codec=args.codec,
        ffmpeg_path=args.ffmpeg_path,
    )

    try:
        stats = run_worker_loop(
            repository=repository,
            generator=generator,
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            max_retries=args.max_retries,
            once=args.once,
            max_jobs=args.max_jobs,
        )
    except KeyboardInterrupt:
        logger.info("Privacy preview worker interrupted by user.")
        return 0

    logger.info("Privacy preview worker stopped: %s", stats)
    return 0 if stats.errors == 0 else 1


def run_worker_loop(
    repository: EventRepository,
    generator: PrivacyPreviewGenerator,
    worker_id: str,
    lease_seconds: int,
    poll_interval_seconds: float,
    max_retries: int,
    once: bool = False,
    max_jobs: Optional[int] = None,
) -> WorkerStats:
    if max_jobs is not None and max_jobs < 0:
        raise ValueError("max_jobs must be greater than or equal to 0")
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be greater than or equal to 0")

    stats = WorkerStats()
    while True:
        if max_jobs is not None and stats.processed >= max_jobs:
            break

        current = process_next_job(
            repository=repository,
            generator=generator,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            max_retries=max_retries,
        )
        stats.add(current)

        if once:
            break
        if current.idle:
            time.sleep(poll_interval_seconds)

    return stats


def process_next_job(
    repository: EventRepository,
    generator: PrivacyPreviewGenerator,
    worker_id: str,
    lease_seconds: int,
    max_retries: int,
) -> WorkerStats:
    stats = WorkerStats()
    job = repository.lease_privacy_preview_job(
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )
    if job is None:
        stats.idle = 1
        logger.debug("No pending privacy preview job found.")
        return stats

    stats.leased = 1
    stats.processed = 1
    job_id = str(job["job_id"])
    event_id = str(job["event_id"])

    try:
        event = repository.get_event(event_id)
        if event is None:
            raise RuntimeError(f"Event does not exist for privacy preview job: {event_id}")
        input_path = str(event["clip_path"])
        output_path = generator.generate(
            input_path=input_path,
            event_id=event_id,
        )
        repository.complete_privacy_preview_job(
            job_id=job_id,
            preview_path=str(output_path),
        )
        stats.completed = 1
        logger.info(
            "Completed privacy preview job: job_id=%s event_id=%s output=%s",
            job_id,
            event_id,
            output_path,
        )
        return stats
    except Exception as exc:
        stats.failed = 1
        logger.exception(
            "Privacy preview job failed: job_id=%s event_id=%s error_type=%s",
            job_id,
            event_id,
            exc.__class__.__name__,
        )
        try:
            repository.fail_privacy_preview_job(
                job_id=job_id,
                error=str(exc),
                max_retries=max_retries,
            )
        except Exception:
            stats.errors += 1
            logger.exception("Failed to record privacy preview job failure: job_id=%s", job_id)
            raise
        return stats


def parse_args() -> argparse.Namespace:
    config_path = _preparse_config_path()
    defaults = _default_arg_values()
    defaults.update(load_config_defaults(config_path))
    parser = build_arg_parser(defaults)
    return parser.parse_args()


def _default_arg_values() -> dict[str, Any]:
    return {
        "config": str(DEFAULT_CONFIG_PATH),
        "queue_db_path": str(DB_PATH),
        "preview_dir": str(PRIVACY_PREVIEW_DIR),
        "worker_id": _default_worker_id(),
        "lease_seconds": 300,
        "poll_interval_seconds": 2.0,
        "max_retries": 1,
        "once": False,
        "max_jobs": None,
        "person_model": DEFAULT_PRIVACY_PREVIEW_MODEL,
        "person_confidence": 0.12,
        "codec": "mp4v",
        "ffmpeg_path": "ffmpeg",
        "log_level": "INFO",
    }


def _preparse_config_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args()
    return str(args.config)


def load_config_defaults(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path).expanduser()
    explicit_config = _was_config_explicitly_requested()
    if not path.exists():
        if explicit_config:
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        return {}
    if not path.is_file():
        raise IsADirectoryError(f"Config path is not a file: {config_path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file is not valid JSON: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    allowed_keys = set(_default_arg_values())
    config: dict[str, Any] = {}
    for key, value in raw_config.items():
        if _is_config_comment_key(key):
            continue
        if key == "privacy_preview_model":
            config["person_model"] = value
            continue
        if key in allowed_keys:
            config[key] = value
    return config


def build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run async silhouette privacy preview worker for fall alert events."
    )
    parser.add_argument("--config", default=defaults["config"])
    parser.add_argument("--queue-db-path", default=defaults["queue_db_path"])
    parser.add_argument("--preview-dir", default=defaults["preview_dir"])
    parser.add_argument("--worker-id", default=defaults["worker_id"])
    parser.add_argument("--lease-seconds", type=int, default=defaults["lease_seconds"])
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=defaults["poll_interval_seconds"],
    )
    parser.add_argument("--max-retries", type=int, default=defaults["max_retries"])
    parser.add_argument("--once", action=argparse.BooleanOptionalAction, default=defaults["once"])
    parser.add_argument("--max-jobs", type=int, default=defaults["max_jobs"])
    parser.add_argument("--person-model", default=defaults["person_model"])
    parser.add_argument(
        "--person-confidence",
        type=float,
        default=defaults["person_confidence"],
    )
    parser.add_argument("--codec", default=defaults["codec"])
    parser.add_argument("--ffmpeg-path", default=defaults["ffmpeg_path"])
    parser.add_argument("--log-level", default=defaults["log_level"])
    return parser


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_worker_id() -> str:
    host = socket.gethostname() or "host"
    return f"privacy_preview_worker_{host}_{os.getpid()}"


def _was_config_explicitly_requested() -> bool:
    import sys

    return any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:])


def _is_config_comment_key(key: str) -> bool:
    return key in CONFIG_COMMENT_KEYS or key.startswith("_")


if __name__ == "__main__":
    raise SystemExit(main())
