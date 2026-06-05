"""
In-memory frame buffer for candidate event windows.

The buffer stores recent frame packets per camera_id. It does not persist frames
and does not copy image arrays by default, so callers should treat returned
packets as read-only.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional


logger = logging.getLogger(__name__)


class EventBufferError(RuntimeError):
    """Raised when the event buffer receives invalid input."""


class EventBuffer:
    """
    Keep recent frame packets in memory for pre/post-event clip construction.

    Args:
        max_seconds: Time window to keep per camera.
        max_frames_per_camera: Optional hard cap for memory control.
    """

    def __init__(
        self,
        max_seconds: float = 10.0,
        max_frames_per_camera: Optional[int] = None,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be greater than 0")
        if max_frames_per_camera is not None and max_frames_per_camera <= 0:
            raise ValueError("max_frames_per_camera must be greater than 0")

        self.max_seconds = float(max_seconds)
        self.max_duration_ms = int(round(self.max_seconds * 1000))
        self.max_frames_per_camera = max_frames_per_camera
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        self._lock = threading.RLock()

    def append(self, packet: Dict[str, Any]) -> None:
        """Append one frame packet and prune old frames for the same camera."""
        camera_id = _packet_camera_id(packet)
        timestamp_ms = _packet_timestamp_ms(packet)

        with self._lock:
            buffer = self._buffers[camera_id]
            buffer.append(packet)
            self._prune_camera(camera_id, newest_timestamp_ms=timestamp_ms)

    def extend(self, packets: List[Dict[str, Any]]) -> None:
        """Append multiple packets."""
        for packet in packets:
            self.append(packet)

    def get_window(
        self,
        camera_id: str,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
    ) -> List[Dict[str, Any]]:
        """
        Return packets whose timestamps are inside [start, end].

        Returned dicts are shallow copies; their "frame" values still reference
        the original arrays.
        """
        if start_timestamp_ms > end_timestamp_ms:
            raise ValueError("start_timestamp_ms must be <= end_timestamp_ms")

        with self._lock:
            buffer = self._buffers.get(camera_id)
            if not buffer:
                return []
            return [
                dict(packet)
                for packet in buffer
                if start_timestamp_ms <= _packet_timestamp_ms(packet) <= end_timestamp_ms
            ]

    def get_recent(
        self,
        camera_id: str,
        seconds: Optional[float] = None,
        before_timestamp_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return recent packets for one camera.

        If before_timestamp_ms is omitted, the latest buffered timestamp is used.
        """
        with self._lock:
            buffer = self._buffers.get(camera_id)
            if not buffer:
                return []

            end_timestamp_ms = (
                int(before_timestamp_ms)
                if before_timestamp_ms is not None
                else _packet_timestamp_ms(buffer[-1])
            )
            duration_ms = (
                self.max_duration_ms
                if seconds is None
                else int(round(float(seconds) * 1000))
            )
            start_timestamp_ms = end_timestamp_ms - max(0, duration_ms)
            return [
                dict(packet)
                for packet in buffer
                if start_timestamp_ms <= _packet_timestamp_ms(packet) <= end_timestamp_ms
            ]

    def get_around(
        self,
        camera_id: str,
        center_timestamp_ms: int,
        pre_seconds: float,
        post_seconds: float,
    ) -> List[Dict[str, Any]]:
        """Return packets around a center timestamp."""
        if pre_seconds < 0 or post_seconds < 0:
            raise ValueError("pre_seconds and post_seconds must be non-negative")
        start_timestamp_ms = int(center_timestamp_ms - pre_seconds * 1000)
        end_timestamp_ms = int(center_timestamp_ms + post_seconds * 1000)
        return self.get_window(camera_id, start_timestamp_ms, end_timestamp_ms)

    def get_all(self, camera_id: str) -> List[Dict[str, Any]]:
        """Return all buffered packets for one camera."""
        with self._lock:
            return [dict(packet) for packet in self._buffers.get(camera_id, [])]

    def clear(self, camera_id: Optional[str] = None) -> None:
        """Clear one camera buffer, or all buffers when camera_id is omitted."""
        with self._lock:
            if camera_id is None:
                self._buffers.clear()
            else:
                self._buffers.pop(camera_id, None)

    def cameras(self) -> List[str]:
        """Return camera IDs that currently have buffered packets."""
        with self._lock:
            return list(self._buffers.keys())

    def size(self, camera_id: Optional[str] = None) -> int:
        """Return buffered packet count for one camera or all cameras."""
        with self._lock:
            if camera_id is not None:
                return len(self._buffers.get(camera_id, []))
            return sum(len(buffer) for buffer in self._buffers.values())

    def get_info(self) -> Dict[str, Any]:
        """Return lightweight buffer state for logging or health checks."""
        with self._lock:
            return {
                "max_seconds": self.max_seconds,
                "max_duration_ms": self.max_duration_ms,
                "max_frames_per_camera": self.max_frames_per_camera,
                "camera_count": len(self._buffers),
                "total_frames": sum(len(buffer) for buffer in self._buffers.values()),
                "per_camera_frames": {
                    camera_id: len(buffer)
                    for camera_id, buffer in self._buffers.items()
                },
            }

    def _prune_camera(self, camera_id: str, newest_timestamp_ms: int) -> None:
        buffer = self._buffers[camera_id]
        min_timestamp_ms = newest_timestamp_ms - self.max_duration_ms

        while buffer and _packet_timestamp_ms(buffer[0]) < min_timestamp_ms:
            buffer.popleft()

        if self.max_frames_per_camera is not None:
            while len(buffer) > self.max_frames_per_camera:
                buffer.popleft()


def _packet_camera_id(packet: Dict[str, Any]) -> str:
    camera_id = packet.get("camera_id")
    if camera_id is None or not str(camera_id).strip():
        raise EventBufferError("frame packet must contain a non-empty camera_id")
    return str(camera_id)


def _packet_timestamp_ms(packet: Dict[str, Any]) -> int:
    if "timestamp_ms" not in packet:
        raise EventBufferError("frame packet must contain timestamp_ms")
    try:
        return int(packet["timestamp_ms"])
    except (TypeError, ValueError) as exc:
        raise EventBufferError("timestamp_ms must be an integer-like value") from exc
