"""
本脚本负责运行独立 VLM worker：从 SQLite 数据库领取待复核任务，读取候选事件的 clip_path，调用 VLM 判断是否摔倒，并把复核结果写回数据库。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import DB_PATH, DEFAULT_VLM_MODEL
from services import event_state
from services.event_repository import EventRepository
from services.video_vlm_verifier import VideoVLMVerifier


logger = logging.getLogger(__name__)

FINAL_VLM_STATUSES = {
    event_state.CONFIRMED_FALL,
    event_state.REJECTED,
    event_state.NEED_HUMAN_REVIEW,
}


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

    repository = EventRepository(args.queue_db_path).initialize()
    verifier = VideoVLMVerifier(
        model_id=args.vlm_model,
        backend=args.vlm_backend,
        max_frames=args.vlm_max_frames,
        max_new_tokens=args.vlm_max_new_tokens,
        temperature=args.vlm_temperature,
    )
    logger.info("Loading VLM model before polling jobs")
    verifier.load()

    try:
        stats = run_worker_loop(
            repository=repository,
            verifier=verifier,
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            max_retries=args.max_retries,
            decision_deadline_seconds=args.decision_deadline_seconds,
            once=args.once,
            max_jobs=args.max_jobs,
        )
    except KeyboardInterrupt:
        logger.info("VLM worker interrupted by user.")
        return 0

    logger.info("VLM worker stopped: %s", stats)
    return 0 if stats.errors == 0 else 1


def run_worker_loop(
    repository: EventRepository,
    verifier: VideoVLMVerifier,
    worker_id: str,
    lease_seconds: int,
    poll_interval_seconds: float,
    max_retries: int,
    decision_deadline_seconds: Optional[float],
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
            verifier=verifier,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            max_retries=max_retries,
            decision_deadline_seconds=decision_deadline_seconds,
        )
        stats.add(current)

        if once:
            break
        if current.idle:
            time.sleep(poll_interval_seconds)

    return stats


def process_next_job(
    repository: EventRepository,
    verifier: VideoVLMVerifier,
    worker_id: str,
    lease_seconds: int,
    max_retries: int,
    decision_deadline_seconds: Optional[float] = None,
) -> WorkerStats:
    stats = WorkerStats()
    job = repository.lease_vlm_job(
        worker_id=worker_id,
        lease_seconds=lease_seconds,
    )
    if job is None:
        stats.idle = 1
        logger.debug("No pending VLM job found.")
        return stats

    stats.leased = 1
    stats.processed = 1
    job_id = str(job["job_id"])
    event_id = str(job["event_id"])

    try:
        event = repository.get_event(event_id)
        if event is None:
            raise RuntimeError(f"Event does not exist for VLM job: {event_id}")

        candidate = event["candidate"]
        clip_path = str(event["clip_path"])
        verification = _verify_with_deadline(
            verifier=verifier,
            candidate=candidate,
            clip_path=clip_path,
            timeout_seconds=decision_deadline_seconds,
        )
        verification = normalize_verification(
            verification=verification,
            event=event,
            job=job,
            worker_id=worker_id,
        )
        final_status = final_status_from_verification(verification)
        repository.complete_vlm_job(
            job_id=job_id,
            verification=verification,
            final_status=final_status,
        )
        stats.completed = 1
        logger.info(
            "Completed VLM job: job_id=%s event_id=%s final_status=%s confidence=%.3f",
            job_id,
            event_id,
            final_status,
            _safe_float(verification.get("confidence"), 0.0),
        )
        return stats
    except Exception as exc:
        stats.failed = 1
        logger.exception(
            "VLM job failed: job_id=%s event_id=%s error_type=%s",
            job_id,
            event_id,
            exc.__class__.__name__,
        )
        try:
            repository.mark_vlm_job_degraded(
                job_id=job_id,
                reason=str(exc),
                failure_status="timeout" if isinstance(exc, TimeoutError) else "failed",
            )
        except Exception:
            stats.errors += 1
            logger.exception("Failed to record VLM job failure: job_id=%s", job_id)
            raise
        return stats


def normalize_verification(
    verification: Dict[str, Any],
    event: Dict[str, Any],
    job: Dict[str, Any],
    worker_id: str,
) -> Dict[str, Any]:
    if not isinstance(verification, dict):
        raise ValueError("VLM verification result must be a dictionary")

    normalized = dict(verification)
    final_status = final_status_from_verification(normalized)
    normalized["result"] = final_status
    normalized.setdefault("camera_id", event.get("camera_id"))

    candidate = event.get("candidate") or {}
    if isinstance(candidate, dict):
        normalized.setdefault("candidate_id", candidate.get("candidate_id"))
        normalized.setdefault("timestamp_ms", candidate.get("timestamp_ms", 0))

    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("vlm_job_id", job.get("job_id"))
    metadata.setdefault("worker_id", worker_id)
    normalized["metadata"] = metadata
    return normalized


def _verify_with_deadline(
    verifier: VideoVLMVerifier,
    candidate: Dict[str, Any],
    clip_path: str,
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    if timeout_seconds is None:
        return verifier.verify(candidate=candidate, clip_path=clip_path)
    if timeout_seconds <= 0:
        raise TimeoutError("VLM decision deadline exceeded")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        verifier.verify,
        candidate=candidate,
        clip_path=clip_path,
    )
    try:
        result = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        # Python cannot interrupt an already-running native/GPU inference thread.
        # Wait before returning so a timed-out job does not keep consuming GPU in
        # the background after the repository has already marked it failed.
        executor.shutdown(wait=True, cancel_futures=True)
        raise TimeoutError("VLM decision deadline exceeded") from exc
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
        return result


def final_status_from_verification(verification: Dict[str, Any]) -> str:
    result = str(verification.get("result", "")).strip()
    if result in FINAL_VLM_STATUSES:
        return result
    return event_state.NEED_HUMAN_REVIEW


def parse_args() -> argparse.Namespace:
    defaults = _default_arg_values()
    parser = build_arg_parser(defaults)
    return parser.parse_args()


def _default_arg_values() -> Dict[str, Any]:
    return {
        "queue_db_path": str(DB_PATH),
        "worker_id": _default_worker_id(),
        "lease_seconds": 300,
        "poll_interval_seconds": 2.0,
        "max_retries": 2,
        "decision_deadline_seconds": 120.0,
        "once": False,
        "max_jobs": None,
        "vlm_model": DEFAULT_VLM_MODEL,
        "vlm_backend": "transformers",
        "vlm_max_frames": 12,
        "vlm_max_new_tokens": 256,
        "vlm_temperature": 0.0,
        "log_level": "INFO",
    }


def build_arg_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run async VLM worker for fall candidate events."
    )
    parser.add_argument("--queue-db-path", default=defaults["queue_db_path"])
    parser.add_argument("--worker-id", default=defaults["worker_id"])
    parser.add_argument("--lease-seconds", type=int, default=defaults["lease_seconds"])
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=defaults["poll_interval_seconds"],
    )
    parser.add_argument("--max-retries", type=int, default=defaults["max_retries"])
    parser.add_argument(
        "--decision-deadline-seconds",
        type=float,
        default=defaults["decision_deadline_seconds"],
    )
    parser.add_argument("--once", action=argparse.BooleanOptionalAction, default=defaults["once"])
    parser.add_argument("--max-jobs", type=int, default=defaults["max_jobs"])

    parser.add_argument("--vlm-model", default=defaults["vlm_model"])
    parser.add_argument(
        "--vlm-backend",
        choices=["transformers", "minicpm_chat"],
        default=defaults["vlm_backend"],
    )
    parser.add_argument("--vlm-max-frames", type=int, default=defaults["vlm_max_frames"])
    parser.add_argument("--vlm-max-new-tokens", type=int, default=defaults["vlm_max_new_tokens"])
    parser.add_argument("--vlm-temperature", type=float, default=defaults["vlm_temperature"])
    parser.add_argument("--log-level", default=defaults["log_level"])
    return parser


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _default_worker_id() -> str:
    host = socket.gethostname() or "host"
    return f"vlm_worker_{host}_{os.getpid()}"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
