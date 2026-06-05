"""
Local video gateway runner.

This script simulates camera streams from local video files, runs YOLO fuzzy
fall candidate detection, optionally verifies candidates with a Video VLM, and
saves confirmed event clips for downstream privacy protection and integrity
modules.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import DEFAULT_VLM_MODEL, EVENT_DIR, TEST_VIDEO_DIR
from services.clip_builder import ClipBuilder
from services.event_buffer import EventBuffer
from services.file_video_source import FileVideoSource
from services.video_vlm_verifier import VideoVLMVerifier
from services.yolo_candidate_detector import YoloCandidateDetector


logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "detection_config.json"
CONFIG_COMMENT_KEYS = {"_注释", "_说明", "_comment", "comment", "参数说明", "使用说明"}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".flv",
    ".webm",
}


@dataclass
class ActiveEvent:
    candidate: Dict[str, Any]
    frames: List[Dict[str, Any]]
    end_timestamp_ms: int
    last_frame_key: Optional[Tuple[int, int]] = None

    def append(self, packet: Dict[str, Any]) -> None:
        key = _frame_key(packet)
        if key == self.last_frame_key:
            return
        self.frames.append(dict(packet))
        self.last_frame_key = key


@dataclass
class PipelineStats:
    videos_seen: int = 0
    videos_processed: int = 0
    frames_read: int = 0
    yolo_candidates: int = 0
    vlm_confirmed: int = 0
    vlm_rejected: int = 0
    vlm_review: int = 0
    clips_saved: int = 0
    errors: int = 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    video_files = scan_video_files(Path(args.video_dir), recursive=args.recursive)
    if args.max_videos is not None:
        video_files = video_files[: args.max_videos]

    stats = PipelineStats(videos_seen=len(video_files))
    if not video_files:
        logger.warning("No video files found: video_dir=%s", args.video_dir)
        return 0

    logger.info("Found %s video file(s) under %s", len(video_files), args.video_dir)

    yolo_detector = YoloCandidateDetector(
        model_path=args.yolo_model,
        device=args.yolo_device,
        imgsz=args.yolo_imgsz,
        conf_threshold=args.yolo_conf,
        candidate_threshold=args.candidate_threshold,
        min_candidate_gap_ms=int(args.cooldown_seconds * 1000),
    )
    vlm_verifier = None
    if not args.skip_vlm:
        vlm_verifier = VideoVLMVerifier(
            model_id=args.vlm_model,
            backend=args.vlm_backend,
            max_frames=args.vlm_max_frames,
            max_new_tokens=args.vlm_max_new_tokens,
            temperature=args.vlm_temperature,
        )

    buffer_seconds = max(
        args.buffer_seconds,
        args.pre_event_seconds + args.post_event_seconds + 2.0,
    )
    event_buffer = EventBuffer(max_seconds=buffer_seconds)
    clip_builder = ClipBuilder(output_dir=args.output_dir)

    for index, video_path in enumerate(video_files, start=1):
        camera_id = f"{args.camera_prefix}_{index:03d}"
        try:
            process_video_file(
                video_path=video_path,
                camera_id=camera_id,
                args=args,
                yolo_detector=yolo_detector,
                vlm_verifier=vlm_verifier,
                event_buffer=event_buffer,
                clip_builder=clip_builder,
                stats=stats,
            )
            stats.videos_processed += 1
        except Exception as exc:
            stats.errors += 1
            logger.exception(
                "Failed to process video: camera_id=%s source_uri=%s error=%s",
                camera_id,
                video_path,
                exc,
            )
        finally:
            event_buffer.clear(camera_id)

    logger.info("Pipeline finished: %s", stats)
    return 0 if stats.errors == 0 else 1


def process_video_file(
    video_path: Path,
    camera_id: str,
    args: argparse.Namespace,
    yolo_detector: YoloCandidateDetector,
    vlm_verifier: Optional[VideoVLMVerifier],
    event_buffer: EventBuffer,
    clip_builder: ClipBuilder,
    stats: PipelineStats,
) -> None:
    logger.info("Processing video: camera_id=%s source_uri=%s", camera_id, video_path)

    source = FileVideoSource(
        camera_id=camera_id,
        source_uri=str(video_path),
        fps_limit=args.fps_limit,
        realtime=args.realtime,
        loop=False,
    )

    active_event: Optional[ActiveEvent] = None
    cooldown_until_ms = -1

    try:
        source.open()
        while True:
            packet = source.read()
            if packet is None:
                break

            stats.frames_read += 1
            timestamp_ms = int(packet["timestamp_ms"])
            event_buffer.append(packet)

            if active_event is not None:
                active_event.append(packet)
                if timestamp_ms >= active_event.end_timestamp_ms:
                    finalize_event(
                        active_event=active_event,
                        vlm_verifier=vlm_verifier,
                        clip_builder=clip_builder,
                        args=args,
                        stats=stats,
                    )
                    cooldown_until_ms = timestamp_ms + int(args.cooldown_seconds * 1000)
                    active_event = None
                continue

            if timestamp_ms < cooldown_until_ms:
                continue

            candidates = yolo_detector.detect(packet)
            if not candidates:
                continue

            stats.yolo_candidates += len(candidates)
            candidate = max(candidates, key=lambda item: item["score"])
            active_event = start_active_event(
                candidate=candidate,
                event_buffer=event_buffer,
                camera_id=camera_id,
                pre_event_seconds=args.pre_event_seconds,
                post_event_seconds=args.post_event_seconds,
            )
            logger.info(
                "YOLO candidate: camera_id=%s candidate_id=%s score=%.3f "
                "timestamp_ms=%s frames_buffered=%s",
                camera_id,
                candidate.get("candidate_id"),
                float(candidate.get("score", 0.0)),
                candidate.get("timestamp_ms"),
                len(active_event.frames),
            )

        if active_event is not None:
            logger.info(
                "Finalizing partial event at end of video: camera_id=%s candidate_id=%s",
                camera_id,
                active_event.candidate.get("candidate_id"),
            )
            finalize_event(
                active_event=active_event,
                vlm_verifier=vlm_verifier,
                clip_builder=clip_builder,
                args=args,
                stats=stats,
            )
    finally:
        source.close()


def start_active_event(
    candidate: Dict[str, Any],
    event_buffer: EventBuffer,
    camera_id: str,
    pre_event_seconds: float,
    post_event_seconds: float,
) -> ActiveEvent:
    candidate_timestamp_ms = int(candidate.get("timestamp_ms", 0))
    start_timestamp_ms = candidate_timestamp_ms - int(pre_event_seconds * 1000)
    end_timestamp_ms = candidate_timestamp_ms + int(post_event_seconds * 1000)
    frames = event_buffer.get_window(camera_id, start_timestamp_ms, candidate_timestamp_ms)
    last_frame_key = _frame_key(frames[-1]) if frames else None
    return ActiveEvent(
        candidate=candidate,
        frames=frames,
        end_timestamp_ms=end_timestamp_ms,
        last_frame_key=last_frame_key,
    )


def finalize_event(
    active_event: ActiveEvent,
    vlm_verifier: Optional[VideoVLMVerifier],
    clip_builder: ClipBuilder,
    args: argparse.Namespace,
    stats: PipelineStats,
) -> None:
    candidate = active_event.candidate
    frames = active_event.frames
    if not frames:
        logger.warning(
            "Skipping event with no frames: candidate_id=%s",
            candidate.get("candidate_id"),
        )
        return

    verification = verify_event(
        candidate=candidate,
        frames=frames,
        vlm_verifier=vlm_verifier,
        skip_vlm=args.skip_vlm,
    )
    result = str(verification.get("result", "need_human_review"))
    confidence = float(verification.get("confidence", 0.0))

    if result == "confirmed_fall":
        stats.vlm_confirmed += 1
    elif result == "rejected":
        stats.vlm_rejected += 1
    else:
        stats.vlm_review += 1

    should_save = should_save_event(
        result=result,
        confidence=confidence,
        min_confidence=args.vlm_confidence_threshold,
        save_review=args.save_review,
        save_rejected=args.save_rejected,
    )

    logger.info(
        "VLM verification: candidate_id=%s result=%s confidence=%.3f save=%s",
        candidate.get("candidate_id"),
        result,
        confidence,
        should_save,
    )

    if not should_save:
        return

    try:
        saved = clip_builder.save_event_clip(
            candidate=candidate,
            verification=verification,
            frame_packets=frames,
            category=result,
        )
        stats.clips_saved += 1
        logger.info(
            "Saved verified event: event_id=%s clip_path=%s metadata_path=%s",
            saved.get("event_id"),
            saved.get("clip_path"),
            saved.get("metadata_path"),
        )
    except Exception as exc:
        stats.errors += 1
        logger.exception(
            "Failed to save event clip: candidate_id=%s error=%s",
            candidate.get("candidate_id"),
            exc,
        )


def verify_event(
    candidate: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    vlm_verifier: Optional[VideoVLMVerifier],
    skip_vlm: bool,
) -> Dict[str, Any]:
    if skip_vlm:
        return {
            "camera_id": candidate.get("camera_id"),
            "candidate_id": candidate.get("candidate_id"),
            "result": "confirmed_fall",
            "confidence": 1.0,
            "reason": "VLM skipped by --skip-vlm; YOLO candidate accepted for debugging.",
            "visible_evidence": ["YOLO candidate was emitted."],
            "raw_response": "",
            "model_id": "skip_vlm",
            "timestamp_ms": candidate.get("timestamp_ms", 0),
            "metadata": {"backend": "skip_vlm"},
            "is_confirmed": True,
        }

    if vlm_verifier is None:
        raise RuntimeError("vlm_verifier is required when skip_vlm is false")

    try:
        return vlm_verifier.verify(
            candidate=candidate,
            frames=[packet["frame"] for packet in frames if packet.get("frame") is not None],
        )
    except Exception as exc:
        logger.exception(
            "VLM verification failed: candidate_id=%s error=%s",
            candidate.get("candidate_id"),
            exc,
        )
        return {
            "camera_id": candidate.get("camera_id"),
            "candidate_id": candidate.get("candidate_id"),
            "result": "need_human_review",
            "confidence": 0.0,
            "reason": f"VLM verification failed: {exc}",
            "visible_evidence": [],
            "raw_response": "",
            "model_id": getattr(vlm_verifier, "model_id", "unknown"),
            "timestamp_ms": candidate.get("timestamp_ms", 0),
            "metadata": {"backend": getattr(vlm_verifier, "backend", "unknown")},
            "is_confirmed": False,
        }


def should_save_event(
    result: str,
    confidence: float,
    min_confidence: float,
    save_review: bool,
    save_rejected: bool,
) -> bool:
    if result == "confirmed_fall":
        return confidence >= min_confidence
    if result == "need_human_review":
        return save_review
    if result == "rejected":
        return save_rejected
    return False


def scan_video_files(video_dir: Path, recursive: bool = False) -> List[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"video_dir does not exist: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video_dir is not a directory: {video_dir}")

    iterator = video_dir.rglob("*") if recursive else video_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    config_path = _preparse_config_path()
    defaults = _default_arg_values()
    config_defaults = load_config_defaults(config_path)
    defaults.update(config_defaults)

    parser = build_arg_parser(defaults)
    return parser.parse_args()


def _preparse_config_path() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args()
    return str(args.config)


def _default_arg_values() -> Dict[str, Any]:
    return {
        "config": str(DEFAULT_CONFIG_PATH),
        "video_dir": str(TEST_VIDEO_DIR),
        "output_dir": str(EVENT_DIR),
        "recursive": False,
        "max_videos": None,
        "camera_prefix": "file_cam",
        "fps_limit": 10.0,
        "realtime": False,
        "pre_event_seconds": 3.0,
        "post_event_seconds": 3.0,
        "buffer_seconds": 10.0,
        "cooldown_seconds": 8.0,
        "yolo_model": "yolo26n-pose.pt",
        "yolo_device": None,
        "yolo_imgsz": 640,
        "yolo_conf": 0.25,
        "candidate_threshold": 0.55,
        "skip_vlm": False,
        "vlm_model": DEFAULT_VLM_MODEL,
        "vlm_backend": "transformers",
        "vlm_max_frames": 12,
        "vlm_max_new_tokens": 256,
        "vlm_temperature": 0.0,
        "vlm_confidence_threshold": 0.6,
        "save_review": False,
        "save_rejected": False,
        "log_level": "INFO",
    }


def load_config_defaults(config_path: str) -> Dict[str, Any]:
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
    config: Dict[str, Any] = {}
    unknown_keys: List[str] = []
    for key, value in raw_config.items():
        if _is_config_comment_key(key):
            continue
        if key not in allowed_keys:
            unknown_keys.append(key)
            continue
        config[key] = value

    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown config option(s) in {config_path}: {joined}")
    return config


def build_arg_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local video fall detection pipeline."
    )
    parser.add_argument("--config", default=defaults["config"])
    parser.add_argument("--video-dir", default=defaults["video_dir"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=defaults["recursive"])
    parser.add_argument("--max-videos", type=int, default=defaults["max_videos"])
    parser.add_argument("--camera-prefix", default=defaults["camera_prefix"])

    parser.add_argument("--fps-limit", type=float, default=defaults["fps_limit"])
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=defaults["realtime"])

    parser.add_argument("--pre-event-seconds", type=float, default=defaults["pre_event_seconds"])
    parser.add_argument("--post-event-seconds", type=float, default=defaults["post_event_seconds"])
    parser.add_argument("--buffer-seconds", type=float, default=defaults["buffer_seconds"])
    parser.add_argument("--cooldown-seconds", type=float, default=defaults["cooldown_seconds"])

    parser.add_argument("--yolo-model", default=defaults["yolo_model"])
    parser.add_argument("--yolo-device", default=defaults["yolo_device"])
    parser.add_argument("--yolo-imgsz", type=int, default=defaults["yolo_imgsz"])
    parser.add_argument("--yolo-conf", type=float, default=defaults["yolo_conf"])
    parser.add_argument("--candidate-threshold", type=float, default=defaults["candidate_threshold"])

    parser.add_argument("--skip-vlm", action=argparse.BooleanOptionalAction, default=defaults["skip_vlm"])
    parser.add_argument("--vlm-model", default=defaults["vlm_model"])
    parser.add_argument(
        "--vlm-backend",
        choices=["transformers", "minicpm_chat"],
        default=defaults["vlm_backend"],
    )
    parser.add_argument("--vlm-max-frames", type=int, default=defaults["vlm_max_frames"])
    parser.add_argument("--vlm-max-new-tokens", type=int, default=defaults["vlm_max_new_tokens"])
    parser.add_argument("--vlm-temperature", type=float, default=defaults["vlm_temperature"])
    parser.add_argument(
        "--vlm-confidence-threshold",
        type=float,
        default=defaults["vlm_confidence_threshold"],
    )
    parser.add_argument("--save-review", action=argparse.BooleanOptionalAction, default=defaults["save_review"])
    parser.add_argument("--save-rejected", action=argparse.BooleanOptionalAction, default=defaults["save_rejected"])

    parser.add_argument("--log-level", default=defaults["log_level"])
    return parser


def _was_config_explicitly_requested() -> bool:
    import sys

    return any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:])


def _is_config_comment_key(key: str) -> bool:
    return key in CONFIG_COMMENT_KEYS or key.startswith("_")


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _frame_key(packet: Dict[str, Any]) -> Tuple[int, int]:
    return int(packet.get("frame_id", -1)), int(packet.get("timestamp_ms", -1))


if __name__ == "__main__":
    raise SystemExit(main())
