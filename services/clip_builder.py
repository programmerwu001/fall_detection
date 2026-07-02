"""
用于保存摔倒检测事件视频和元数据，既支持保存尚未经过 VLM 复核的 candidates 候选事件，也支持保存 confirmed_fall、rejected、need_human_review 等复核后的事件。

Confirmed event clip writer.

This module writes selected frame packets to a video file and stores metadata
for downstream privacy protection, encryption, hashing, and anti-deletion
modules. It does not perform those downstream tasks itself.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]

try:
    from config import (
        DISABLED_DEBUG_EVENT_DIR,
        EVENT_DIR,
        PRIVATE_EVENT_DIR,
        DEFAULT_SAVE_DEBUG_RAW_EVENT_COPY,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    EVENT_DIR = Path("data") / "events"
    PRIVATE_EVENT_DIR = Path("data") / "private_events"
    DISABLED_DEBUG_EVENT_DIR = Path("data") / "debug_events_disabled"
    DEFAULT_SAVE_DEBUG_RAW_EVENT_COPY = True


logger = logging.getLogger(__name__)


class ClipBuilderError(RuntimeError):
    """Raised when an event clip cannot be written."""


@dataclass
class SavedClip:
    """Metadata for a saved event clip."""

    event_id: str
    camera_id: str
    clip_path: str
    metadata_path: str
    frame_count: int
    fps: float
    width: int
    height: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    duration_ms: int
    category: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "clip_path": self.clip_path,
            "metadata_path": self.metadata_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "start_timestamp_ms": self.start_timestamp_ms,
            "end_timestamp_ms": self.end_timestamp_ms,
            "duration_ms": self.duration_ms,
            "category": self.category,
            **self.extra,
        }


class ClipBuilder:
    """
    Save event frame packets to mp4 clips and JSON metadata.

    Args:
        output_dir: Root directory for event files.
        codec: OpenCV fourcc codec. "mp4v" is broadly available.
        extension: Output video extension.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        internal_output_dir: Optional[str | Path] = None,
        disabled_debug_dir: Optional[str | Path] = None,
        save_debug_raw_event_copy: bool = DEFAULT_SAVE_DEBUG_RAW_EVENT_COPY,
        codec: str = "mp4v",
        extension: str = ".mp4",
    ) -> None:
        if not codec or len(codec) != 4:
            raise ValueError("codec must be a 4-character fourcc string")
        if not extension.startswith("."):
            extension = "." + extension

        self.output_dir = Path(output_dir) if output_dir else Path(EVENT_DIR)
        self.internal_output_dir = (
            Path(internal_output_dir) if internal_output_dir else self.output_dir
        )
        self.disabled_debug_dir = (
            Path(disabled_debug_dir) if disabled_debug_dir else Path(DISABLED_DEBUG_EVENT_DIR)
        )
        self.save_debug_raw_event_copy = bool(save_debug_raw_event_copy)
        self.codec = codec
        self.extension = extension

        if not self.save_debug_raw_event_copy:
            self._move_existing_debug_files()

    def save_event_clip(
        self,
        candidate: Dict[str, Any],
        verification: Optional[Dict[str, Any]],
        frame_packets: Sequence[Dict[str, Any]],
        event_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a verified event clip and metadata.

        Callers decide whether rejected events should be saved. This method
        writes whatever category it is given.
        """
        if not frame_packets:
            raise ClipBuilderError("frame_packets must not be empty")

        camera_id = _camera_id_from(candidate, frame_packets)
        category = category or _category_from_verification(verification)
        private_clip_dir = self._event_dir(camera_id, self.internal_output_dir)
        debug_clip_dir = self._event_dir(camera_id, self.output_dir)
        private_clip_dir.mkdir(parents=True, exist_ok=True)
        if self.save_debug_raw_event_copy:
            debug_clip_dir.mkdir(parents=True, exist_ok=True)

        event_stem = self._next_event_stem(private_clip_dir, debug_clip_dir)
        event_id = event_id or self._make_event_id(camera_id, private_clip_dir.name, event_stem)

        clip_path = private_clip_dir / f"{event_stem}{self.extension}"
        metadata_path = private_clip_dir / f"{event_stem}.json"
        debug_clip_path = debug_clip_dir / f"{event_stem}{self.extension}"
        debug_metadata_path = debug_clip_dir / f"{event_stem}.json"

        fps = _select_fps(frame_packets)
        width, height = _select_frame_size(frame_packets)
        start_timestamp_ms = _timestamp_ms(frame_packets[0])
        end_timestamp_ms = _timestamp_ms(frame_packets[-1])
        duration_ms = max(0, end_timestamp_ms - start_timestamp_ms)

        written_frames = self._write_video(
            clip_path=clip_path,
            frame_packets=frame_packets,
            fps=fps,
            width=width,
            height=height,
        )

        saved = SavedClip(
            event_id=event_id,
            camera_id=camera_id,
            clip_path=str(clip_path),
            metadata_path=str(metadata_path),
            frame_count=written_frames,
            fps=fps,
            width=width,
            height=height,
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            duration_ms=duration_ms,
            category=category,
            extra={
                "source_uri": frame_packets[0].get("source_uri", ""),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "candidate": candidate,
                "verification": verification,
                "privacy_status": "raw_unprotected",
                "integrity_status": "not_hashed",
                "retention_status": "pending_manifest",
            },
        )

        self._write_metadata(metadata_path, saved.to_dict())
        result = saved.to_dict()
        if self.save_debug_raw_event_copy:
            if clip_path.exists():
                shutil.copy2(clip_path, debug_clip_path)
            self._write_metadata(
                debug_metadata_path,
                _debug_metadata(saved.to_dict()),
            )
            result["debug_clip_path"] = str(debug_clip_path)
            result["debug_metadata_path"] = str(debug_metadata_path)
        logger.info(
            "Saved event clip: event_id=%s camera_id=%s category=%s debug_copy=%s",
            event_id,
            camera_id,
            category,
            self.save_debug_raw_event_copy,
        )
        return result

    def _write_video(
        self,
        clip_path: Path,
        frame_packets: Sequence[Dict[str, Any]],
        fps: float,
        width: int,
        height: int,
    ) -> int:
        if cv2 is None:
            raise ClipBuilderError("OpenCV is required to write event clips.")
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise ClipBuilderError("Failed to open video writer")

        written_frames = 0
        try:
            for packet in frame_packets:
                frame = packet.get("frame")
                if frame is None:
                    continue
                frame = _normalize_frame(frame, width=width, height=height)
                writer.write(frame)
                written_frames += 1
        finally:
            writer.release()

        if written_frames == 0:
            raise ClipBuilderError("No frames were written to the event clip")
        if clip_path.suffix.lower() == ".mp4" and clip_path.exists():
            self._transcode_browser_mp4(clip_path)
        return written_frames

    def _transcode_browser_mp4(self, clip_path: Path) -> None:
        tmp_path = Path(str(clip_path) + ".tmp.mp4")
        if tmp_path.exists():
            tmp_path.unlink()
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tmp_path.replace(clip_path)
        except FileNotFoundError:
            logger.warning("ffmpeg is not available; saved clip may not play in browsers")
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning(
                "Failed to transcode clip for browser playback: %s",
                exc.__class__.__name__,
            )
            if tmp_path.exists():
                tmp_path.unlink()

    def _write_metadata(self, metadata_path: Path, metadata: Dict[str, Any]) -> None:
        try:
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=True, indent=2, default=str)
        except OSError as exc:
            raise ClipBuilderError("Failed to write event metadata") from exc

    def _event_dir(self, camera_id: str, root: Path) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        return root / _safe_name(camera_id) / today

    def _next_event_stem(self, *clip_dirs: Path) -> str:
        max_index = 0
        pattern = re.compile(r"^event_(\d+)$")
        for clip_dir in clip_dirs:
            if not clip_dir.exists():
                continue
            for path in clip_dir.glob(f"event_*{self.extension}"):
                match = pattern.match(path.stem)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        return f"event_{max_index + 1}"

    def _make_event_id(self, camera_id: str, date_dir: str, event_stem: str) -> str:
        raw = f"{camera_id}_{date_dir}_{event_stem}"
        return _safe_name(raw)[:180]

    def _move_existing_debug_files(self) -> None:
        if not self.output_dir.exists():
            return
        for path in sorted(self.output_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".mp4", ".json"}:
                continue
            relative = path.relative_to(self.output_dir)
            target = _unique_path(self.disabled_debug_dir / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))


def _camera_id_from(
    candidate: Dict[str, Any], frame_packets: Sequence[Dict[str, Any]]
) -> str:
    camera_id = candidate.get("camera_id") or frame_packets[0].get("camera_id")
    if camera_id is None or not str(camera_id).strip():
        raise ClipBuilderError("camera_id is required in candidate or frame packets")
    return str(camera_id)


def _select_fps(frame_packets: Sequence[Dict[str, Any]], default_fps: float = 10.0) -> float:
    for packet in frame_packets:
        fps = packet.get("fps")
        if fps:
            try:
                fps_value = float(fps)
                if fps_value > 0:
                    return fps_value
            except (TypeError, ValueError):
                continue
    return default_fps


def _select_frame_size(frame_packets: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    for packet in frame_packets:
        width = packet.get("width")
        height = packet.get("height")
        if width and height:
            width_int = int(width)
            height_int = int(height)
            if width_int > 0 and height_int > 0:
                return width_int, height_int

    frame = frame_packets[0].get("frame")
    if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
        return int(frame.shape[1]), int(frame.shape[0])

    raise ClipBuilderError("Unable to determine frame width and height")


def _timestamp_ms(packet: Dict[str, Any]) -> int:
    try:
        return int(packet.get("timestamp_ms", 0))
    except (TypeError, ValueError):
        return 0


def _category_from_verification(verification: Optional[Dict[str, Any]]) -> str:
    if verification is None:
        return "candidates"
    return str(verification.get("result") or "unknown")


def _normalize_frame(frame: Any, width: int, height: int) -> Any:
    if not hasattr(frame, "shape"):
        raise ClipBuilderError("frame must be an OpenCV/numpy image array")

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) >= 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) < 3 or frame.shape[2] < 3:
        raise ClipBuilderError("unsupported frame channel format")

    current_height, current_width = frame.shape[:2]
    if current_width != width or current_height != height:
        frame = cv2.resize(frame, (width, height))
    return frame


def _safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or "unknown"


def _debug_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "event_id",
        "camera_id",
        "frame_count",
        "fps",
        "width",
        "height",
        "start_timestamp_ms",
        "end_timestamp_ms",
        "duration_ms",
        "category",
        "created_at",
        "privacy_status",
        "integrity_status",
        "retention_status",
    }
    debug = {key: metadata.get(key) for key in allowed if key in metadata}
    debug["candidate"] = _without_private_fields(metadata.get("candidate") or {})
    verification = metadata.get("verification")
    debug["verification"] = (
        _without_private_fields(verification) if isinstance(verification, dict) else verification
    )
    return debug


def _without_private_fields(value: Any) -> Any:
    blocked = {
        "clip_path",
        "debug_clip_path",
        "metadata_path",
        "debug_metadata_path",
        "media_url",
        "source_uri",
        "raw_clip_path",
        "staging_plaintext_path",
        "evidence_path",
    }
    if isinstance(value, dict):
        return {
            key: _without_private_fields(item)
            for key, item in value.items()
            if key not in blocked
        }
    if isinstance(value, list):
        return [_without_private_fields(item) for item in value]
    return value


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
