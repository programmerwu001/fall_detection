"""
High-recall YOLO candidate detector for possible fall events.

This module is the first stage of the fall pipeline. It is intentionally fuzzy:
it should find possible falls cheaply and pass only short candidate clips to a
stronger verifier such as a small Video VLM.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


class YoloCandidateDetectorError(RuntimeError):
    """Raised when the YOLO candidate detector cannot be initialized or run."""


@dataclass
class FallCandidate:
    """A fuzzy fall candidate emitted by the YOLO stage."""

    camera_id: str
    candidate_id: str
    frame_id: int
    timestamp_ms: int
    track_id: int
    score: float
    bbox: List[float]
    reason: Dict[str, Any]
    source_uri: str
    model_name: str
    width: int
    height: int
    keypoints: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "candidate_id": self.candidate_id,
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "track_id": self.track_id,
            "score": self.score,
            "bbox": self.bbox,
            "reason": self.reason,
            "source_uri": self.source_uri,
            "model_name": self.model_name,
            "width": self.width,
            "height": self.height,
            "keypoints": self.keypoints,
        }


@dataclass
class _TrackState:
    track_id: int
    bbox: List[float]
    center_y: float
    last_timestamp_ms: int
    low_pose_started_ms: Optional[int] = None
    last_candidate_ms: Optional[int] = None


class YoloCandidateDetector:
    """
    Use Ultralytics YOLO to generate possible fall candidates.

    The detector supports YOLO pose models and plain person detection models.
    Pose models are preferred because shoulder/hip keypoints make the fuzzy
    score much more meaningful.
    """

    def __init__(
        self,
        model_path: str = "yolo26n-pose.pt",
        device: Optional[Any] = None,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        candidate_threshold: float = 0.4,
        person_class_id: Optional[int] = 0,
        track_iou_threshold: float = 0.35,
        track_ttl_ms: int = 3000,
        min_candidate_gap_ms: int = 1500,
        drop_ratio_threshold: float = 0.06,
        drop_window_ms: int = 1500,
        min_low_pose_ms: int = 500,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be provided")
        if imgsz <= 0:
            raise ValueError("imgsz must be greater than 0")
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        if not 0 <= candidate_threshold <= 1:
            raise ValueError("candidate_threshold must be between 0 and 1")

        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.candidate_threshold = candidate_threshold
        self.person_class_id = person_class_id
        self.track_iou_threshold = track_iou_threshold
        self.track_ttl_ms = track_ttl_ms
        self.min_candidate_gap_ms = min_candidate_gap_ms
        self.drop_ratio_threshold = drop_ratio_threshold
        self.drop_window_ms = drop_window_ms
        self.min_low_pose_ms = min_low_pose_ms

        self._model: Any = None
        self._tracks: Dict[int, _TrackState] = {}
        self._next_track_id = 1

    def load(self) -> "YoloCandidateDetector":
        """Load the YOLO model lazily."""
        if self._model is not None:
            return self

        YOLO = _import_ultralytics_yolo()
        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:  # pragma: no cover - depends on model runtime
            raise YoloCandidateDetectorError(
                "Failed to load YOLO model. Check model_path, dependencies, "
                "and whether the weights are available locally."
            ) from exc

        logger.info("Loaded YOLO candidate detector: model_path=%s", self.model_path)
        return self

    def detect(self, frame_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return fuzzy fall candidates as dictionaries."""
        return [candidate.to_dict() for candidate in self.detect_candidates(frame_packet)]

    def detect_candidates(self, frame_packet: Dict[str, Any]) -> List[FallCandidate]:
        """Run YOLO on one frame packet and return fuzzy fall candidates."""
        self.load()

        frame = frame_packet.get("frame")
        if frame is None:
            raise ValueError("frame_packet must contain a 'frame' value")

        camera_id = str(frame_packet.get("camera_id", "unknown"))
        frame_id = int(frame_packet.get("frame_id", 0))
        timestamp_ms = int(frame_packet.get("timestamp_ms", 0))
        source_uri = str(frame_packet.get("source_uri", ""))
        width = int(frame_packet.get("width") or _frame_width(frame))
        height = int(frame_packet.get("height") or _frame_height(frame))

        try:
            results = self._predict(frame)
        except Exception as exc:  # pragma: no cover - depends on model runtime
            raise YoloCandidateDetectorError("YOLO inference failed") from exc

        if not results:
            self._drop_stale_tracks(timestamp_ms)
            return []

        detections = self._extract_person_detections(results[0])
        self._assign_track_ids(detections)

        candidates: List[FallCandidate] = []
        for detection in detections:
            candidate = self._score_detection(
                detection=detection,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_uri=source_uri,
                width=width,
                height=height,
            )
            if candidate is not None:
                candidates.append(candidate)

        self._drop_stale_tracks(timestamp_ms)
        return candidates

    def detect_one(self, frame_packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the highest-scoring candidate for this frame, if any."""
        candidates = self.detect(frame_packet)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item["score"])

    def get_state(self) -> Dict[str, Any]:
        """Return lightweight debug state for logging or health endpoints."""
        return {
            "model_path": self.model_path,
            "loaded": self._model is not None,
            "tracked_person_count": len(self._tracks),
            "next_track_id": self._next_track_id,
            "candidate_threshold": self.candidate_threshold,
        }

    def reset_state(self) -> None:
        """Clear cross-frame tracking state at a known stream boundary."""
        self._tracks.clear()
        self._next_track_id = 1

    def _predict(self, frame: Any) -> Sequence[Any]:
        kwargs: Dict[str, Any] = {
            "source": frame,
            "imgsz": self.imgsz,
            "conf": self.conf_threshold,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.person_class_id is not None:
            kwargs["classes"] = [self.person_class_id]
        return self._model.predict(**kwargs)

    def _extract_person_detections(self, result: Any) -> List[Dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        xyxy = _tensor_to_list(boxes.xyxy)
        confs = _tensor_to_list(getattr(boxes, "conf", []))
        classes = _tensor_to_list(getattr(boxes, "cls", []))

        keypoints_xy: List[Any] = []
        keypoints_conf: List[Any] = []
        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and getattr(keypoints, "xy", None) is not None:
            keypoints_xy = _tensor_to_list(keypoints.xy)
            if getattr(keypoints, "conf", None) is not None:
                keypoints_conf = _tensor_to_list(keypoints.conf)

        detections: List[Dict[str, Any]] = []
        for index, bbox in enumerate(xyxy):
            cls_id = int(classes[index]) if index < len(classes) else 0
            if self.person_class_id is not None and cls_id != self.person_class_id:
                continue

            conf = float(confs[index]) if index < len(confs) else 0.0
            person_keypoints = _merge_keypoints(
                keypoints_xy[index] if index < len(keypoints_xy) else [],
                keypoints_conf[index] if index < len(keypoints_conf) else [],
            )
            detections.append(
                {
                    "bbox": [float(v) for v in bbox],
                    "confidence": conf,
                    "class_id": cls_id,
                    "keypoints": person_keypoints,
                    "track_id": None,
                }
            )
        return detections

    def _assign_track_ids(self, detections: List[Dict[str, Any]]) -> None:
        used_track_ids = set()
        for detection in detections:
            best_track_id = None
            best_iou = 0.0
            for track_id, state in self._tracks.items():
                if track_id in used_track_ids:
                    continue
                iou = _bbox_iou(detection["bbox"], state.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.track_iou_threshold:
                detection["track_id"] = best_track_id
                used_track_ids.add(best_track_id)
            else:
                detection["track_id"] = self._next_track_id
                used_track_ids.add(self._next_track_id)
                self._next_track_id += 1

    def _score_detection(
        self,
        detection: Dict[str, Any],
        camera_id: str,
        frame_id: int,
        timestamp_ms: int,
        source_uri: str,
        width: int,
        height: int,
    ) -> Optional[FallCandidate]:
        bbox = detection["bbox"]
        track_id = int(detection["track_id"])
        confidence = float(detection["confidence"])
        keypoints = detection.get("keypoints", [])

        x1, y1, x2, y2 = bbox
        bbox_width = max(1.0, x2 - x1)
        bbox_height = max(1.0, y2 - y1)
        center_y = (y1 + y2) / 2.0
        aspect_ratio = bbox_width / bbox_height
        center_y_ratio = center_y / max(1.0, float(height))

        previous = self._tracks.get(track_id)
        drop_ratio = 0.0
        center_drop = False
        if previous is not None:
            delta_ms = timestamp_ms - previous.last_timestamp_ms
            if 0 < delta_ms <= self.drop_window_ms:
                drop_ratio = (center_y - previous.center_y) / max(1.0, float(height))
                center_drop = drop_ratio >= self.drop_ratio_threshold

        torso_angle = _torso_angle_from_keypoints(keypoints)
        angle_score = 0.0
        if torso_angle is not None:
            angle_score = _clamp((torso_angle - 35.0) / 45.0, 0.0, 1.0)

        aspect_score = _clamp((aspect_ratio - 0.75) / 0.75, 0.0, 1.0)
        drop_score = _clamp(drop_ratio / 0.18, 0.0, 1.0)
        low_position_score = 1.0 if center_y_ratio >= 0.55 else 0.0

        lying_by_pose = torso_angle is not None and torso_angle >= 55.0
        lying_by_bbox = aspect_ratio >= 1.15
        low_pose_now = lying_by_pose or lying_by_bbox

        low_pose_started_ms = None
        if low_pose_now:
            low_pose_started_ms = (
                previous.low_pose_started_ms
                if previous and previous.low_pose_started_ms is not None
                else timestamp_ms
            )

        persistence_ms = (
            timestamp_ms - low_pose_started_ms
            if low_pose_started_ms is not None
            else 0
        )
        persistence_score = (
            1.0 if persistence_ms >= self.min_low_pose_ms else 0.0
        )

        fuzzy_score = (
            0.25 * angle_score
            + 0.20 * aspect_score
            + 0.25 * drop_score
            + 0.15 * low_position_score
            + 0.15 * persistence_score
        )
        fuzzy_score = _clamp(fuzzy_score * (0.65 + 0.35 * confidence), 0.0, 1.0)

        last_candidate_ms = previous.last_candidate_ms if previous else None
        should_emit = fuzzy_score >= self.candidate_threshold
        if last_candidate_ms is not None:
            should_emit = (
                should_emit
                and timestamp_ms - last_candidate_ms >= self.min_candidate_gap_ms
            )

        self._tracks[track_id] = _TrackState(
            track_id=track_id,
            bbox=bbox,
            center_y=center_y,
            last_timestamp_ms=timestamp_ms,
            low_pose_started_ms=low_pose_started_ms,
            last_candidate_ms=timestamp_ms if should_emit else last_candidate_ms,
        )

        if not should_emit:
            return None

        reason = {
            "confidence": round(confidence, 4),
            "torso_angle_deg": round(torso_angle, 2) if torso_angle is not None else None,
            "aspect_ratio": round(aspect_ratio, 4),
            "center_y_ratio": round(center_y_ratio, 4),
            "drop_ratio": round(drop_ratio, 4),
            "lying_by_pose": lying_by_pose,
            "lying_by_bbox": lying_by_bbox,
            "center_drop": center_drop,
            "low_pose_persistence_ms": persistence_ms,
            "score_parts": {
                "angle": round(angle_score, 4),
                "aspect": round(aspect_score, 4),
                "drop": round(drop_score, 4),
                "low_position": round(low_position_score, 4),
                "persistence": round(persistence_score, 4),
            },
        }

        candidate_id = f"{camera_id}_{track_id}_{frame_id}_{timestamp_ms}"
        return FallCandidate(
            camera_id=camera_id,
            candidate_id=candidate_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            track_id=track_id,
            score=round(fuzzy_score, 4),
            bbox=[round(float(v), 2) for v in bbox],
            reason=reason,
            source_uri=source_uri,
            model_name=self.model_path,
            width=width,
            height=height,
            keypoints=keypoints,
        )

    def _drop_stale_tracks(self, timestamp_ms: int) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if timestamp_ms - state.last_timestamp_ms > self.track_ttl_ms
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)


def _import_ultralytics_yolo() -> Any:
    try:
        from ultralytics import YOLO

        return YOLO
    except Exception as first_error:
        project_root = Path(__file__).resolve().parents[1]
        third_party_repo = project_root / "third_party" / "ultralytics"
        if third_party_repo.exists():
            repo_path = str(third_party_repo)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            try:
                from ultralytics import YOLO

                return YOLO
            except Exception as second_error:
                raise YoloCandidateDetectorError(
                    "Failed to import Ultralytics YOLO from installed packages "
                    "or third_party/ultralytics. Install its dependencies first."
                ) from second_error

        raise YoloCandidateDetectorError(
            "Ultralytics is not installed and third_party/ultralytics was not found."
        ) from first_error


def _tensor_to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _merge_keypoints(points: Any, confidences: Any) -> List[List[float]]:
    merged: List[List[float]] = []
    if not points:
        return merged

    for index, point in enumerate(points):
        if point is None or len(point) < 2:
            continue
        conf = (
            float(confidences[index])
            if confidences is not None and index < len(confidences)
            else 0.0
        )
        merged.append([float(point[0]), float(point[1]), conf])
    return merged


def _torso_angle_from_keypoints(keypoints: List[List[float]]) -> Optional[float]:
    # COCO keypoint indices: 5/6 shoulders, 11/12 hips.
    if len(keypoints) <= 12:
        return None

    left_shoulder = _valid_point(keypoints[5])
    right_shoulder = _valid_point(keypoints[6])
    left_hip = _valid_point(keypoints[11])
    right_hip = _valid_point(keypoints[12])

    shoulder = _midpoint(left_shoulder, right_shoulder)
    hip = _midpoint(left_hip, right_hip)
    if shoulder is None or hip is None:
        return None

    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    if dx == 0 and dy == 0:
        return None

    angle_from_vertical = abs(math.degrees(math.atan2(dx, dy)))
    return _clamp(angle_from_vertical, 0.0, 90.0)


def _valid_point(point: Sequence[float], min_conf: float = 0.2) -> Optional[Tuple[float, float]]:
    if len(point) < 2:
        return None
    if len(point) >= 3 and float(point[2]) < min_conf:
        return None
    x, y = float(point[0]), float(point[1])
    if x <= 0 and y <= 0:
        return None
    return x, y


def _midpoint(
    left: Optional[Tuple[float, float]], right: Optional[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _frame_width(frame: Any) -> int:
    return int(frame.shape[1]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else 0


def _frame_height(frame: Any) -> int:
    return int(frame.shape[0]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else 0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
