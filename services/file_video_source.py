"""
Local file-backed video source for camera stream simulation.

This module only handles video input. It intentionally does not implement
fall detection, encryption, web APIs, or database persistence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime in open()
    cv2 = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class FileVideoSourceError(RuntimeError):
    """Raised when a local video source cannot be opened or read correctly."""


class FileVideoSource:
    """
    Wrap a local video file as a frame-by-frame stream source.

    Args:
        camera_id: Logical camera identifier used by downstream services.
        source_uri: Local path to the video file.
        fps_limit: Optional maximum output FPS. Frames are skipped to meet it.
        realtime: If True, sleep between emitted frames according to timestamps.
        loop: If True, rewind to the beginning when the file ends.
    """

    DEFAULT_FPS = 25.0
    SUPPORTED_EXTENSIONS = {
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

    def __init__(
        self,
        camera_id: str,
        source_uri: str,
        fps_limit: Optional[float] = None,
        realtime: bool = False,
        loop: bool = False,
    ) -> None:
        if not camera_id or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        if not source_uri or not str(source_uri).strip():
            raise ValueError("source_uri must be a non-empty local video path")
        if fps_limit is not None and fps_limit <= 0:
            raise ValueError("fps_limit must be greater than 0 when provided")

        self.camera_id = camera_id
        self.source_uri = str(source_uri)
        self.fps_limit = float(fps_limit) if fps_limit is not None else None
        self.realtime = bool(realtime)
        self.loop = bool(loop)

        self._capture: Any = None
        self._opened = False

        self._width = 0
        self._height = 0
        self._source_fps = self.DEFAULT_FPS
        self._effective_fps = self.DEFAULT_FPS
        self._frame_count = 0
        self._duration_ms = 0.0

        self._packet_frame_id = 0
        self._source_frame_index = 0
        self._loop_count = 0
        self._timestamp_offset_ms = 0.0
        self._next_emit_timestamp_ms = 0.0
        self._stream_started_at: Optional[float] = None

    def open(self) -> "FileVideoSource":
        """Open the configured local video file and load basic metadata."""
        if self._opened:
            logger.debug("Video source already opened: camera_id=%s", self.camera_id)
            return self

        if cv2 is None:
            raise FileVideoSourceError(
                "OpenCV is not installed. Install opencv-python to use FileVideoSource."
            )

        video_path = Path(self.source_uri).expanduser()
        if not video_path.exists():
            raise FileNotFoundError("Video file does not exist")
        if not video_path.is_file():
            raise FileVideoSourceError("Video source is not a file")
        if video_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                "Video extension is not in the common supported list: camera_id=%s suffix=%s",
                self.camera_id,
                video_path.suffix.lower(),
            )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise FileVideoSourceError("Failed to open video file")

        self._capture = capture
        self._opened = True
        self._load_metadata()
        self._reset_runtime_state()

        logger.info(
            "Opened file video source: camera_id=%s width=%s height=%s "
            "source_fps=%.3f output_fps=%.3f frame_count=%s realtime=%s loop=%s",
            self.camera_id,
            self._width,
            self._height,
            self._source_fps,
            self._effective_fps,
            self._frame_count,
            self.realtime,
            self.loop,
        )
        return self

    def read(self) -> Optional[Dict[str, Any]]:
        """
        Read the next emitted frame packet.

        Returns:
            A dict containing camera_id, frame_id, timestamp_ms, frame, width,
            height, fps, and source_uri. Returns None when the file ends and
            loop=False.
        """
        self._ensure_opened()

        while True:
            ok, frame = self._capture.read()
            if not ok:
                if self.loop and self._source_frame_index > 0:
                    self._rewind_for_loop()
                    continue

                logger.info(
                    "Reached end of video source: camera_id=%s",
                    self.camera_id,
                )
                return None

            source_frame_index = self._source_frame_index
            self._source_frame_index += 1

            timestamp_ms = self._timestamp_offset_ms + self._frame_timestamp_ms(
                source_frame_index
            )

            if not self._should_emit(timestamp_ms):
                continue

            self._sleep_until_realtime(timestamp_ms)

            height, width = frame.shape[:2]
            packet = {
                "camera_id": self.camera_id,
                "frame_id": self._packet_frame_id,
                "timestamp_ms": int(round(timestamp_ms)),
                "frame": frame,
                "width": self._width or width,
                "height": self._height or height,
                "fps": self._effective_fps,
                "source_uri": self.source_uri,
                # Extra fields are useful for debugging skipped frames and loops.
                "source_fps": self._source_fps,
                "source_frame_index": source_frame_index,
                "loop_count": self._loop_count,
            }
            self._packet_frame_id += 1
            return packet

    def close(self) -> None:
        """Release the underlying OpenCV video capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        if self._opened:
            logger.info(
                "Closed file video source: camera_id=%s",
                self.camera_id,
            )

        self._opened = False

    def get_info(self) -> Dict[str, Any]:
        """Return basic metadata and current runtime state."""
        return {
            "camera_id": self.camera_id,
            "source_uri": self.source_uri,
            "opened": self._opened,
            "width": self._width,
            "height": self._height,
            "source_fps": self._source_fps,
            "fps": self._effective_fps,
            "fps_limit": self.fps_limit,
            "frame_count": self._frame_count,
            "duration_ms": int(round(self._duration_ms)),
            "realtime": self.realtime,
            "loop": self.loop,
            "emitted_frame_count": self._packet_frame_id,
            "loop_count": self._loop_count,
        }

    @property
    def is_opened(self) -> bool:
        """Whether the underlying video capture is currently opened."""
        return self._opened

    def __enter__(self) -> "FileVideoSource":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _load_metadata(self) -> None:
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            logger.warning(
                "Video FPS is missing or invalid; using default %.1f FPS: camera_id=%s",
                self.DEFAULT_FPS,
                self.camera_id,
            )
            fps = self.DEFAULT_FPS

        self._source_fps = fps
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._duration_ms = (
            (self._frame_count / self._source_fps) * 1000.0
            if self._frame_count > 0
            else 0.0
        )

        if self.fps_limit is None:
            self._effective_fps = self._source_fps
        else:
            self._effective_fps = min(self.fps_limit, self._source_fps)

    def _reset_runtime_state(self) -> None:
        self._packet_frame_id = 0
        self._source_frame_index = 0
        self._loop_count = 0
        self._timestamp_offset_ms = 0.0
        self._next_emit_timestamp_ms = 0.0
        self._stream_started_at = None

    def _ensure_opened(self) -> None:
        if not self._opened or self._capture is None:
            raise FileVideoSourceError(
                "Video source is not opened. Call open() before read()."
            )

    def _frame_timestamp_ms(self, source_frame_index: int) -> float:
        opencv_timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if opencv_timestamp > 0:
            return opencv_timestamp
        return (source_frame_index / self._source_fps) * 1000.0

    def _should_emit(self, timestamp_ms: float) -> bool:
        if self.fps_limit is None or self.fps_limit >= self._source_fps:
            return True

        if timestamp_ms + 1e-6 < self._next_emit_timestamp_ms:
            return False

        interval_ms = 1000.0 / self._effective_fps
        while self._next_emit_timestamp_ms <= timestamp_ms + 1e-6:
            self._next_emit_timestamp_ms += interval_ms
        return True

    def _sleep_until_realtime(self, timestamp_ms: float) -> None:
        if not self.realtime:
            return

        now = time.monotonic()
        if self._stream_started_at is None:
            self._stream_started_at = now - (timestamp_ms / 1000.0)

        target_time = self._stream_started_at + (timestamp_ms / 1000.0)
        delay = target_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _rewind_for_loop(self) -> None:
        loop_duration_ms = self._duration_ms
        if loop_duration_ms <= 0:
            loop_duration_ms = (self._source_frame_index / self._source_fps) * 1000.0

        self._timestamp_offset_ms += loop_duration_ms
        self._source_frame_index = 0
        self._loop_count += 1

        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise FileVideoSourceError("Failed to rewind video source for loop")

        logger.debug(
            "Looped file video source: camera_id=%s loop_count=%s",
            self.camera_id,
            self._loop_count,
        )
