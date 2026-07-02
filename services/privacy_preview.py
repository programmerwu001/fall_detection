"""Generate caregiver-safe silhouette previews from private event clips."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]

try:
    from config import DEFAULT_PRIVACY_PREVIEW_MODEL, PRIVACY_PREVIEW_DIR
except Exception:  # pragma: no cover - standalone import fallback
    DEFAULT_PRIVACY_PREVIEW_MODEL = "yolo11n-seg.pt"
    PRIVACY_PREVIEW_DIR = Path("data") / "privacy_previews"


logger = logging.getLogger(__name__)


class PrivacyPreviewError(RuntimeError):
    """Raised when a privacy preview cannot be generated or validated."""


@dataclass(frozen=True)
class PersonRegion:
    """One detected person mask or fallback bounding box."""

    mask: Any = None
    box: Optional[Tuple[float, float, float, float]] = None


class PersonDetector(Protocol):
    def detect(self, frame: Any) -> Sequence[PersonRegion]:
        """Return all detected people for one frame."""


def apply_person_silhouettes(
    frame: Any,
    regions: Iterable[PersonRegion],
    color: Tuple[int, int, int] = (6, 6, 6),
    edge_feather_pixels: int = 1,
) -> Any:
    """Return a copy of frame with every detected person region painted opaque dark."""
    if frame is None or not hasattr(frame, "shape"):
        raise PrivacyPreviewError("frame must be an OpenCV/numpy image array")
    output = frame.copy()
    height, width = output.shape[:2]
    fill = np.array(color, dtype=output.dtype)
    for region in regions:
        if region.mask is not None:
            mask = _normalize_mask(region.mask, width=width, height=height)
            if mask.any():
                privacy_mask = _repair_privacy_mask(
                    mask,
                    frame_height=height,
                    edge_feather_pixels=edge_feather_pixels,
                )
                _paint_mask(output, privacy_mask, fill, edge_feather_pixels=0)
            continue
        if region.box is not None:
            x1, y1, x2, y2 = _clip_box(region.box, width=width, height=height)
            if x2 > x1 and y2 > y1:
                mask = np.zeros((height, width), dtype=bool)
                mask[y1:y2, x1:x2] = True
                privacy_mask = _repair_privacy_mask(
                    mask,
                    frame_height=height,
                    edge_feather_pixels=edge_feather_pixels,
                )
                _paint_mask(output, privacy_mask, fill, edge_feather_pixels=0)
    return output


class YoloPersonDetector:
    """Ultralytics-backed person mask detector, with padded boxes as fallback."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_PRIVACY_PREVIEW_MODEL,
        confidence: float = 0.12,
        mask_threshold: float = 0.35,
        box_padding_ratio: float = 0.12,
        inference_size: int = 960,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.mask_threshold = float(mask_threshold)
        self.box_padding_ratio = float(box_padding_ratio)
        self.inference_size = int(inference_size)
        self._model = None

    def load(self) -> "YoloPersonDetector":
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise PrivacyPreviewError("ultralytics is required for person detection") from exc
            self._model = YOLO(self.model_path)
        return self

    def detect(self, frame: Any) -> Sequence[PersonRegion]:
        if self._model is None:
            self.load()
        if frame is None or not hasattr(frame, "shape"):
            raise PrivacyPreviewError("frame must be an OpenCV/numpy image array")

        results = self._model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            imgsz=self.inference_size,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", []))
        classes = _to_numpy(getattr(boxes, "cls", []))
        masks = None
        result_masks = getattr(result, "masks", None)
        if result_masks is not None and getattr(result_masks, "data", None) is not None:
            masks = _to_numpy(result_masks.data)

        height, width = frame.shape[:2]
        regions: list[PersonRegion] = []
        for index, box in enumerate(xyxy):
            if classes.size and int(classes[index]) != 0:
                continue
            mask = None
            if masks is not None and index < len(masks):
                mask = _normalize_mask(
                    masks[index],
                    width=width,
                    height=height,
                    threshold=self.mask_threshold,
                )
            if mask is not None and mask.any():
                regions.append(PersonRegion(mask=mask))
            else:
                regions.append(
                    PersonRegion(
                        box=_pad_box(
                            tuple(float(value) for value in box[:4]),
                            width=width,
                            height=height,
                            ratio=self.box_padding_ratio,
                        )
                    )
                )
        return regions


class PrivacyPreviewGenerator:
    """Generate privacy_preview.mp4 files under a dedicated preview root."""

    def __init__(
        self,
        preview_root: str | Path = PRIVACY_PREVIEW_DIR,
        detector: Optional[PersonDetector] = None,
        codec: str = "mp4v",
        ffmpeg_path: str | Path = "ffmpeg",
        silhouette_color: Tuple[int, int, int] = (6, 6, 6),
    ) -> None:
        if not codec or len(codec) != 4:
            raise ValueError("codec must be a 4-character fourcc string")
        if not str(ffmpeg_path).strip():
            raise ValueError("ffmpeg_path must be non-empty")
        self.preview_root = Path(preview_root)
        self.detector = detector if detector is not None else YoloPersonDetector()
        self.codec = codec
        self.ffmpeg_path = _resolve_ffmpeg_path(ffmpeg_path)
        self.silhouette_color = silhouette_color

    def generate(self, input_path: str | Path, event_id: str) -> Path:
        """Generate one privacy preview from the private source clip."""
        if cv2 is None:
            raise PrivacyPreviewError("OpenCV is required to generate privacy previews")
        source = Path(input_path)
        if not source.exists() or not source.is_file():
            raise PrivacyPreviewError(f"Input clip does not exist: {source}")

        output_dir = self.preview_root / _safe_name(event_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "privacy_preview.mp4"
        tmp_path = output_dir / f"privacy_preview.{uuid.uuid4().hex}.tmp.mp4"

        try:
            self._write_preview(source, tmp_path)
            self._transcode_browser_mp4(tmp_path)
            _validate_video_file(tmp_path)
            tmp_path.replace(final_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if isinstance(exc, PrivacyPreviewError):
                raise
            raise PrivacyPreviewError(str(exc)) from exc
        return final_path

    def _write_preview(self, source: Path, tmp_path: Path) -> None:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise PrivacyPreviewError(f"Failed to open input clip: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 10.0
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise PrivacyPreviewError(f"Input clip has no readable frames: {source}")

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*self.codec),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            writer.release()
            raise PrivacyPreviewError("Failed to open privacy preview writer")

        written = 0
        last_gray: Optional[np.ndarray] = None
        last_mask: Optional[np.ndarray] = None
        last_box: Optional[tuple[int, int, int, int]] = None
        try:
            while ok and frame is not None:
                gray = _frame_gray(frame)
                detected_regions = list(self.detector.detect(frame))
                detected_mask = _regions_to_privacy_mask(
                    detected_regions,
                    width=width,
                    height=height,
                )
                predicted_mask = None
                if last_gray is not None and last_mask is not None:
                    predicted_mask = _propagate_mask(last_gray, gray, last_mask)

                if detected_mask is not None and detected_mask.any():
                    output_mask = detected_mask
                    history_mask = predicted_mask if predicted_mask is not None else last_mask
                    if history_mask is not None and _mask_is_suspiciously_small_or_different(
                        detected_mask,
                        history_mask,
                    ):
                        output_mask = _repair_privacy_mask(
                            detected_mask | history_mask,
                            frame_height=height,
                            edge_feather_pixels=0,
                        )
                elif predicted_mask is not None and predicted_mask.any():
                    output_mask = _repair_privacy_mask(
                        predicted_mask,
                        frame_height=height,
                        edge_feather_pixels=0,
                    )
                elif last_mask is not None and last_mask.any():
                    output_mask = _conservative_history_mask(
                        last_mask,
                        last_box=last_box,
                        width=width,
                        height=height,
                    )
                else:
                    output_mask = np.ones((height, width), dtype=bool)

                output = frame.copy()
                _paint_mask(
                    output,
                    output_mask,
                    np.array(self.silhouette_color, dtype=output.dtype),
                    edge_feather_pixels=0,
                )
                writer.write(output)
                if output_mask is not None and output_mask.any() and not output_mask.all():
                    last_gray = gray
                    last_mask = output_mask.copy()
                    last_box = _mask_to_box(output_mask)
                written += 1
                ok, frame = capture.read()
        finally:
            capture.release()
            writer.release()

        if written == 0:
            raise PrivacyPreviewError("No frames were written to privacy preview")

    def _transcode_browser_mp4(self, clip_path: Path) -> None:
        tmp_path = Path(str(clip_path) + ".h264.tmp.mp4")
        if tmp_path.exists():
            tmp_path.unlink()
        command = [
            self.ffmpeg_path,
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
            if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
                raise PrivacyPreviewError("ffmpeg produced an empty privacy preview")
            tmp_path.replace(clip_path)
        except FileNotFoundError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise PrivacyPreviewError(
                "ffmpeg is required to transcode privacy preview for browser playback"
            ) from exc
        except PrivacyPreviewError:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        except (OSError, subprocess.CalledProcessError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            reason = exc.__class__.__name__
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                reason = exc.stderr.decode("utf-8", errors="replace").strip() or reason
            raise PrivacyPreviewError(
                f"Failed to transcode privacy preview for browser playback: {reason}"
            ) from exc


def _resolve_ffmpeg_path(ffmpeg_path: str | Path) -> str:
    value = str(ffmpeg_path)
    path = Path(value)
    if path.parent != Path(".") or path.name.lower() not in {"ffmpeg", "ffmpeg.exe"}:
        return value

    found = shutil.which(value)
    if found:
        return found

    for candidate in (
        Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
        Path(sys.prefix) / "bin" / "ffmpeg",
    ):
        if candidate.exists():
            return str(candidate)
    return value


def _paint_mask(
    output: Any,
    mask: Any,
    fill: np.ndarray,
    edge_feather_pixels: int,
) -> None:
    privacy_mask = np.asarray(mask, dtype=bool)
    if edge_feather_pixels:
        privacy_mask = _dilate_mask_radius(privacy_mask, max(0, int(edge_feather_pixels)))
    if privacy_mask.any():
        output[privacy_mask] = fill


def _repair_privacy_mask(
    mask: Any,
    frame_height: int,
    edge_feather_pixels: int,
) -> np.ndarray:
    repaired = np.asarray(mask, dtype=bool)
    if repaired.ndim > 2:
        repaired = repaired.squeeze().astype(bool)
    if not repaired.any():
        return repaired

    close_radius = _adaptive_close_radius(frame_height)
    if close_radius > 0:
        repaired = _erode_mask_radius(_dilate_mask_radius(repaired, close_radius), close_radius)
    repaired = _fill_mask_holes(repaired)

    dilation_pixels = _adaptive_dilation_pixels(
        frame_height,
        edge_feather_pixels=edge_feather_pixels,
    )
    return _dilate_mask_radius(repaired, dilation_pixels)


def _adaptive_close_radius(frame_height: int) -> int:
    if frame_height < 32:
        return 1
    return max(1, min(4, int(round(frame_height * 0.003))))


def _adaptive_dilation_pixels(frame_height: int, edge_feather_pixels: int = 0) -> int:
    base = 0 if frame_height < 32 else max(2, min(8, int(round(frame_height * 0.006))))
    return base + max(0, int(edge_feather_pixels))


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    background = ~mask.astype(bool)
    if not background.any():
        return mask.astype(bool)

    if cv2 is not None and hasattr(cv2, "connectedComponents"):
        try:
            component_count, labels = cv2.connectedComponents(
                background.astype(np.uint8),
                connectivity=4,
            )
        except TypeError:
            component_count, labels = cv2.connectedComponents(background.astype(np.uint8), 4)
        if component_count <= 1:
            return mask.astype(bool)
        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        outside = np.isin(labels, border_labels)
        holes = background & ~outside
        return mask | holes

    outside = np.zeros(mask.shape, dtype=bool)
    outside[0, :] = background[0, :]
    outside[-1, :] = background[-1, :]
    outside[:, 0] |= background[:, 0]
    outside[:, -1] |= background[:, -1]
    while True:
        grown = _dilate_mask(outside) & background
        if np.array_equal(grown, outside):
            break
        outside = grown
    return mask | (background & ~outside)


def _dilate_mask_radius(mask: np.ndarray, pixels: int) -> np.ndarray:
    grown = mask.astype(bool)
    for _ in range(max(0, int(pixels))):
        grown = _dilate_mask(grown)
    return grown


def _erode_mask_radius(mask: np.ndarray, pixels: int) -> np.ndarray:
    eroded = mask.astype(bool)
    for _ in range(max(0, int(pixels))):
        eroded = ~_dilate_mask(~eroded)
    return eroded


def _regions_to_privacy_mask(
    regions: Sequence[PersonRegion],
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    combined = np.zeros((height, width), dtype=bool)
    has_region = False
    for region in regions:
        if region.mask is not None:
            mask = _normalize_mask(region.mask, width=width, height=height)
            if mask.any():
                combined |= mask
                has_region = True
                continue
        if region.box is not None:
            x1, y1, x2, y2 = _clip_box(region.box, width=width, height=height)
            if x2 > x1 and y2 > y1:
                combined[y1:y2, x1:x2] = True
                has_region = True

    if not has_region:
        return None
    return _repair_privacy_mask(
        combined,
        frame_height=height,
        edge_feather_pixels=1,
    )


def _frame_gray(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if cv2 is not None and hasattr(cv2, "cvtColor") and hasattr(cv2, "COLOR_BGR2GRAY"):
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return np.rint(array.mean(axis=2)).astype(np.uint8)


def _propagate_mask(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_mask: np.ndarray,
) -> Optional[np.ndarray]:
    if (
        cv2 is None
        or not hasattr(cv2, "goodFeaturesToTrack")
        or not hasattr(cv2, "calcOpticalFlowPyrLK")
        or previous_gray.shape[:2] != current_gray.shape[:2]
        or previous_mask.shape[:2] != previous_gray.shape[:2]
        or not previous_mask.any()
    ):
        return None

    height, width = previous_mask.shape[:2]
    feature_mask = previous_mask.astype(np.uint8) * 255
    min_distance = max(3, min(height, width) // 80)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=feature_mask,
        maxCorners=120,
        qualityLevel=0.01,
        minDistance=min_distance,
    )
    if points is None or len(points) < 3:
        return None

    criteria = None
    if hasattr(cv2, "TERM_CRITERIA_EPS") and hasattr(cv2, "TERM_CRITERIA_COUNT"):
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    flow_kwargs = {
        "winSize": (21, 21),
        "maxLevel": 3,
    }
    if criteria is not None:
        flow_kwargs["criteria"] = criteria
    try:
        next_points, status, _error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            **flow_kwargs,
        )
    except Exception:
        return None
    if next_points is None or status is None:
        return None

    valid = status.reshape(-1).astype(bool)
    if valid.sum() < 3 or valid.sum() / len(points) < 0.35:
        return None

    deltas = next_points.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
    median_delta = np.median(deltas, axis=0)
    if not np.all(np.isfinite(median_delta)):
        return None
    deviations = np.linalg.norm(deltas - median_delta, axis=1)
    if float(np.median(deviations)) > max(6.0, min(height, width) * 0.08):
        return None

    dx, dy = float(median_delta[0]), float(median_delta[1])
    if abs(dx) > width * 0.45 or abs(dy) > height * 0.45:
        return None
    return _shift_mask(previous_mask, dx=dx, dy=dy)


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    shift_x = int(round(dx))
    shift_y = int(round(dy))
    shifted = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape[:2]

    src_x1 = max(0, -shift_x)
    src_x2 = min(width, width - shift_x)
    dst_x1 = max(0, shift_x)
    dst_x2 = min(width, width + shift_x)
    src_y1 = max(0, -shift_y)
    src_y2 = min(height, height - shift_y)
    dst_y1 = max(0, shift_y)
    dst_y2 = min(height, height + shift_y)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return shifted
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def _mask_is_suspiciously_small_or_different(
    current_mask: np.ndarray,
    history_mask: np.ndarray,
) -> bool:
    current_area = int(np.count_nonzero(current_mask))
    history_area = int(np.count_nonzero(history_mask))
    if current_area == 0 or history_area == 0:
        return False
    if current_area < history_area * 0.65:
        return True

    intersection = int(np.count_nonzero(current_mask & history_mask))
    union = int(np.count_nonzero(current_mask | history_mask))
    if union == 0:
        return False
    iou = intersection / union
    return iou < 0.20 and current_area < history_area * 1.25


def _conservative_history_mask(
    last_mask: np.ndarray,
    last_box: Optional[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> np.ndarray:
    extra_pixels = _adaptive_dilation_pixels(height, edge_feather_pixels=2)
    mask = _dilate_mask_radius(last_mask, extra_pixels)
    if last_box is not None:
        x1, y1, x2, y2 = last_box
        pad = extra_pixels * 2
        box_mask = np.zeros((height, width), dtype=bool)
        box_mask[
            max(0, y1 - pad) : min(height, y2 + pad),
            max(0, x1 - pad) : min(width, x2 + pad),
        ] = True
        mask |= box_mask
    return mask


def _mask_to_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    grown = np.zeros((height, width), dtype=bool)
    for y_offset in range(3):
        for x_offset in range(3):
            grown |= padded[y_offset : y_offset + height, x_offset : x_offset + width]
    return grown


def _normalize_mask(mask: Any, width: int, height: int, threshold: float = 0.5) -> Any:
    array = _to_numpy(mask)
    if array.ndim > 2:
        array = array.squeeze()
    if array.shape[:2] != (height, width):
        if cv2 is None:
            raise PrivacyPreviewError("OpenCV is required to resize person masks")
        array = cv2.resize(
            array.astype("float32"),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    return array > threshold


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _clip_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def _pad_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
    ratio: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * max(0.0, ratio)
    pad_y = (y2 - y1) * max(0.0, ratio)
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    )


def _validate_video_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise PrivacyPreviewError("Privacy preview output file is empty")
    if cv2 is None:
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise PrivacyPreviewError("Privacy preview output cannot be opened")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise PrivacyPreviewError("Privacy preview output has no readable frames")
    finally:
        capture.release()


def _safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._")
    return value or "unknown"
