"""Build post-run metrics reports from saved fall-event outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.event_repository import EventRepository


REPORT_JSON_NAME = "metrics_summary.json"
REPORT_MARKDOWN_NAME = "metrics_summary.md"
TEMPORAL_THRESHOLDS_MS = (1000, 2000)


@dataclass
class LabelRecord:
    source_uri: str
    event_start_ms: int
    event_end_ms: int


@dataclass
class DetectionRecord:
    event_id: str
    source_uri: str
    timestamp_ms: int
    category: str


@dataclass
class VideoLabelRecord:
    source_uri: str
    has_fall: bool


@dataclass
class VideoPredictionRecord:
    source_uri: str
    prediction: str


def build_metrics_report(
    event_dir: Path | str,
    queue_db_path: Optional[Path | str] = None,
    labels_path: Optional[Path | str] = None,
    video_labels_path: Optional[Path | str] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a structured metrics report for saved event metadata."""
    event_dir_path = Path(event_dir)
    metadata = load_event_metadata(event_dir_path)

    report: Dict[str, Any] = {
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "event_dir": str(event_dir_path),
        "events": summarize_events(metadata),
        "clips": summarize_clips(metadata),
        "yolo": summarize_yolo(metadata),
        "vlm": summarize_vlm(metadata),
        "queue": summarize_queue(queue_db_path),
        "label_evaluation": evaluate_labels(metadata, labels_path),
        "video_label_evaluation": evaluate_video_labels(
            metadata,
            video_labels_path,
        ),
    }
    report["presentation_highlights"] = build_presentation_highlights(report)
    return report


def load_event_metadata(event_dir: Path) -> List[Dict[str, Any]]:
    if not event_dir.exists():
        return []

    events: List[Dict[str, Any]] = []
    for path in sorted(event_dir.rglob("*.json")):
        if path.name in {REPORT_JSON_NAME}:
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        if not loaded.get("event_id") or not loaded.get("category"):
            continue
        loaded.setdefault("metadata_path", str(path))
        events.append(loaded)
    return events


def summarize_events(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, int] = {}
    cameras = set()
    sources = set()
    for event in metadata:
        category = str(event.get("category") or "unknown")
        by_category[category] = by_category.get(category, 0) + 1
        camera_id = str(event.get("camera_id") or "").strip()
        source_uri = str(event.get("source_uri") or "").strip()
        if camera_id:
            cameras.add(camera_id)
        if source_uri:
            sources.add(source_uri)

    return {
        "total": len(metadata),
        "by_category": dict(sorted(by_category.items())),
        "cameras": len(cameras),
        "sources": len(sources),
    }


def summarize_clips(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    duration_ms = [_safe_float(event.get("duration_ms")) for event in metadata]
    duration_ms = [value for value in duration_ms if value is not None]
    frame_counts = [_safe_int(event.get("frame_count")) for event in metadata]
    frame_counts = [value for value in frame_counts if value is not None]
    fps_values = [_safe_float(event.get("fps")) for event in metadata]
    fps_values = [value for value in fps_values if value is not None and value > 0]

    return {
        "duration_seconds": {
            "total": _round(sum(duration_ms) / 1000.0),
            "average": _round(_average(duration_ms, scale=1000.0)),
            "max": _round((max(duration_ms) / 1000.0) if duration_ms else 0.0),
        },
        "frames": {
            "total": int(sum(frame_counts)),
            "average": _round(_average(frame_counts)),
        },
        "fps": {
            "average": _round(_average(fps_values)),
        },
    }


def summarize_yolo(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores: List[float] = []
    for event in metadata:
        candidate = event.get("candidate")
        if not isinstance(candidate, dict):
            continue
        score = _safe_float(candidate.get("score"))
        if score is not None:
            scores.append(score)

    return {
        "candidates": len(
            [event for event in metadata if isinstance(event.get("candidate"), dict)]
        ),
        "average_score": _round(_average(scores)),
        "max_score": _round(max(scores) if scores else 0.0),
    }


def summarize_vlm(metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result_counts = {
        "confirmed_fall": 0,
        "rejected": 0,
        "need_human_review": 0,
    }
    confidences: List[float] = []
    verified_events = 0

    for event in metadata:
        verification = event.get("verification")
        if not isinstance(verification, dict):
            continue
        verified_events += 1
        result = str(verification.get("result") or event.get("category") or "")
        if result in result_counts:
            result_counts[result] += 1
        confidence = _safe_float(verification.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)

    return {
        "verified_events": verified_events,
        **result_counts,
        "average_confidence": _round(_average(confidences)),
    }


def summarize_queue(queue_db_path: Optional[Path | str]) -> Dict[str, Any]:
    if queue_db_path is None:
        return {
            "available": False,
            "db_path": None,
            "jobs": {},
            "reason": "queue db was not requested",
        }

    db_path = Path(queue_db_path)
    if not db_path.exists():
        return {
            "available": False,
            "db_path": str(db_path),
            "jobs": {},
            "reason": "queue db does not exist",
        }

    try:
        jobs = EventRepository(db_path).get_queue_stats()
    except Exception as exc:
        return {
            "available": False,
            "db_path": str(db_path),
            "jobs": {},
            "reason": str(exc),
        }

    return {
        "available": True,
        "db_path": str(db_path),
        "jobs": jobs,
    }


def evaluate_labels(
    metadata: Sequence[Dict[str, Any]],
    labels_path: Optional[Path | str],
) -> Dict[str, Any]:
    if labels_path is None:
        return {
            "available": False,
            "reason": "labels file was not provided",
        }

    path = Path(labels_path)
    if not path.exists():
        return {
            "available": False,
            "labels_path": str(path),
            "reason": "labels file does not exist",
        }

    labels = load_labels(path)
    detections = build_detection_records(metadata)
    matches = match_labels_to_detections(labels, detections)
    true_positives = len(matches)
    false_positives = max(0, len(detections) - true_positives)
    false_negatives = max(0, len(labels) - true_positives)
    precision = _ratio(true_positives, len(detections))
    recall = _ratio(true_positives, len(labels))
    f1 = _ratio(2 * precision * recall, precision + recall)
    errors = [abs(detection.timestamp_ms - label.event_start_ms) for label, detection in matches]

    return {
        "available": True,
        "labels_path": str(path),
        "labels": len(labels),
        "detections": len(detections),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "start_time_accuracy": {
            f"within_{threshold}ms": _round(
                _ratio(
                    sum(1 for error in errors if error <= threshold),
                    len(labels),
                )
            )
            for threshold in TEMPORAL_THRESHOLDS_MS
        },
        "start_time_error_ms": {
            "mean_abs": _round(_average(errors)),
            "max_abs": _round(max(errors) if errors else 0.0),
        },
    }


def load_labels(path: Path) -> List[LabelRecord]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels: List[LabelRecord] = []
        for row in reader:
            source_uri = str(row.get("source_uri") or "").strip()
            if not source_uri:
                continue
            labels.append(
                LabelRecord(
                    source_uri=source_uri,
                    event_start_ms=int(float(row.get("event_start_ms") or 0)),
                    event_end_ms=int(float(row.get("event_end_ms") or 0)),
                )
            )
    return labels


def evaluate_video_labels(
    metadata: Sequence[Dict[str, Any]],
    video_labels_path: Optional[Path | str],
) -> Dict[str, Any]:
    if video_labels_path is None:
        return {
            "available": False,
            "reason": "video labels file was not provided",
        }

    path = Path(video_labels_path)
    if not path.exists():
        return {
            "available": False,
            "video_labels_path": str(path),
            "reason": "video labels file does not exist",
        }

    labels = load_video_labels(path)
    predictions = build_video_prediction_records(metadata)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    confirmed_positive_predictions = 0
    review_positive_predictions = 0
    rejected_negative_predictions = 0
    pending_sources = 0

    for label in labels:
        matching_predictions = [
            prediction
            for prediction in predictions
            if _source_matches(label.source_uri, prediction.source_uri)
        ]
        predicted_fall = any(
            prediction.prediction in {"confirmed_fall", "need_human_review"}
            for prediction in matching_predictions
        )
        if any(
            prediction.prediction == "confirmed_fall"
            for prediction in matching_predictions
        ):
            confirmed_positive_predictions += 1
        if any(
            prediction.prediction == "need_human_review"
            for prediction in matching_predictions
        ):
            review_positive_predictions += 1
        if any(prediction.prediction == "rejected" for prediction in matching_predictions):
            rejected_negative_predictions += 1
        if any(prediction.prediction == "candidates" for prediction in matching_predictions):
            pending_sources += 1

        if label.has_fall and predicted_fall:
            true_positives += 1
        elif label.has_fall and not predicted_fall:
            false_negatives += 1
        elif not label.has_fall and predicted_fall:
            false_positives += 1
        else:
            true_negatives += 1

    positives = true_positives + false_negatives
    negatives = true_negatives + false_positives
    total = len(labels)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, positives)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return {
        "available": True,
        "video_labels_path": str(path),
        "labels": total,
        "positive_labels": positives,
        "negative_labels": negatives,
        "detected_sources": len(
            {
                Path(prediction.source_uri).name
                for prediction in predictions
                if prediction.prediction in {"confirmed_fall", "need_human_review"}
            }
        ),
        "confirmed_positive_predictions": confirmed_positive_predictions,
        "review_positive_predictions": review_positive_predictions,
        "rejected_negative_predictions": rejected_negative_predictions,
        "pending_sources": pending_sources,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "accuracy": _round(_ratio(true_positives + true_negatives, total)),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
    }


def load_video_labels(path: Path) -> List[VideoLabelRecord]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels: List[VideoLabelRecord] = []
        for row in reader:
            source_uri = str(row.get("source_uri") or "").strip()
            if not source_uri:
                continue
            labels.append(
                VideoLabelRecord(
                    source_uri=source_uri,
                    has_fall=_parse_bool(row.get("has_fall")),
                )
            )
    return labels


def build_video_prediction_records(
    metadata: Sequence[Dict[str, Any]],
) -> List[VideoPredictionRecord]:
    predictions: List[VideoPredictionRecord] = []
    for event in metadata:
        category = str(event.get("category") or "").strip()
        if category not in {
            "confirmed_fall",
            "need_human_review",
            "rejected",
            "candidates",
        }:
            continue
        source_uri = str(event.get("source_uri") or "").strip()
        if not source_uri:
            candidate = event.get("candidate")
            if isinstance(candidate, dict):
                source_uri = str(candidate.get("source_uri") or "").strip()
        if not source_uri:
            continue
        predictions.append(
            VideoPredictionRecord(
                source_uri=source_uri,
                prediction=category,
            )
        )
    return predictions


def build_detection_records(metadata: Sequence[Dict[str, Any]]) -> List[DetectionRecord]:
    detections: List[DetectionRecord] = []
    for event in metadata:
        category = str(event.get("category") or "")
        if category == "rejected":
            continue
        candidate = event.get("candidate")
        if not isinstance(candidate, dict):
            continue
        timestamp_ms = _safe_int(candidate.get("timestamp_ms"))
        source_uri = str(event.get("source_uri") or candidate.get("source_uri") or "").strip()
        if timestamp_ms is None or not source_uri:
            continue
        detections.append(
            DetectionRecord(
                event_id=str(event.get("event_id") or ""),
                source_uri=source_uri,
                timestamp_ms=timestamp_ms,
                category=category or "unknown",
            )
        )
    return detections


def match_labels_to_detections(
    labels: Sequence[LabelRecord],
    detections: Sequence[DetectionRecord],
    tolerance_ms: int = max(TEMPORAL_THRESHOLDS_MS),
) -> List[Tuple[LabelRecord, DetectionRecord]]:
    matches: List[Tuple[LabelRecord, DetectionRecord]] = []
    used_detection_indexes = set()

    for label in labels:
        best_index: Optional[int] = None
        best_error: Optional[int] = None
        for index, detection in enumerate(detections):
            if index in used_detection_indexes:
                continue
            if not _source_matches(label.source_uri, detection.source_uri):
                continue
            error = abs(detection.timestamp_ms - label.event_start_ms)
            if error > tolerance_ms:
                continue
            if best_error is None or error < best_error:
                best_error = error
                best_index = index
        if best_index is None:
            continue
        used_detection_indexes.add(best_index)
        matches.append((label, detections[best_index]))

    return matches


def build_presentation_highlights(report: Dict[str, Any]) -> List[str]:
    highlights: List[str] = []
    events = report["events"]
    clips = report["clips"]
    yolo = report["yolo"]
    vlm = report["vlm"]
    candidate_count = int(events.get("by_category", {}).get("candidates", 0))
    confirmed_count = int(events.get("by_category", {}).get("confirmed_fall", 0))
    avg_duration = clips["duration_seconds"]["average"]

    if candidate_count:
        highlights.append(
            f"{candidate_count} candidate clip(s) were queued for VLM review."
        )
    if confirmed_count:
        highlights.append(f"{confirmed_count} confirmed fall event(s) were retained.")
    if yolo["candidates"]:
        highlights.append(
            f"YOLO produced {yolo['candidates']} candidate event(s) with average score {yolo['average_score']}."
        )
    if vlm["verified_events"]:
        highlights.append(
            f"VLM reviewed {vlm['verified_events']} event(s) with average confidence {vlm['average_confidence']}."
        )
    if avg_duration:
        highlights.append(f"Average saved clip duration was {avg_duration} second(s).")

    label_evaluation = report["label_evaluation"]
    if label_evaluation.get("available"):
        highlights.append(
            "Label evaluation: "
            f"precision={label_evaluation['precision']}, "
            f"recall={label_evaluation['recall']}, "
            f"F1={label_evaluation['f1']}."
        )
    video_label_evaluation = report["video_label_evaluation"]
    if video_label_evaluation.get("available"):
        highlights.append(
            "Video-level evaluation: "
            f"accuracy={video_label_evaluation['accuracy']}, "
            f"precision={video_label_evaluation['precision']}, "
            f"recall={video_label_evaluation['recall']}, "
            f"F1={video_label_evaluation['f1']}."
        )
    return highlights


def render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Fall Edge Gateway Metrics Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Event directory: `{report['event_dir']}`",
        "",
        "## Run Output Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Events | {report['events']['total']} |",
        f"| Cameras | {report['events']['cameras']} |",
        f"| Sources | {report['events']['sources']} |",
        f"| Total clip seconds | {report['clips']['duration_seconds']['total']} |",
        f"| Average clip seconds | {report['clips']['duration_seconds']['average']} |",
        f"| Total saved frames | {report['clips']['frames']['total']} |",
        "",
        "## Event Categories",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in report["events"]["by_category"].items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## YOLO / VLM",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| YOLO candidates | {report['yolo']['candidates']} |",
            f"| Average YOLO score | {report['yolo']['average_score']} |",
            f"| VLM verified events | {report['vlm']['verified_events']} |",
            f"| VLM confirmed | {report['vlm']['confirmed_fall']} |",
            f"| VLM rejected | {report['vlm']['rejected']} |",
            f"| VLM human review | {report['vlm']['need_human_review']} |",
            f"| Average VLM confidence | {report['vlm']['average_confidence']} |",
            "",
            "## VLM Queue",
            "",
        ]
    )

    queue = report["queue"]
    if queue.get("available"):
        lines.extend(["| Status | Jobs |", "|---|---:|"])
        for status, count in queue["jobs"].items():
            lines.append(f"| {status} | {count} |")
    else:
        lines.append(f"- Queue metrics unavailable: {queue.get('reason')}")

    lines.extend(["", "## Label Evaluation", ""])
    label_evaluation = report["label_evaluation"]
    if label_evaluation.get("available"):
        lines.extend(
            [
                "| Metric | Value |",
                "|---|---:|",
                f"| Labels | {label_evaluation['labels']} |",
                f"| Detections | {label_evaluation['detections']} |",
                f"| True positives | {label_evaluation['true_positives']} |",
                f"| False positives | {label_evaluation['false_positives']} |",
                f"| False negatives | {label_evaluation['false_negatives']} |",
                f"| Precision | {label_evaluation['precision']} |",
                f"| Recall | {label_evaluation['recall']} |",
                f"| F1 | {label_evaluation['f1']} |",
                f"| Start time accuracy within 1000ms | {label_evaluation['start_time_accuracy']['within_1000ms']} |",
                f"| Start time accuracy within 2000ms | {label_evaluation['start_time_accuracy']['within_2000ms']} |",
                f"| Mean absolute start time error ms | {label_evaluation['start_time_error_ms']['mean_abs']} |",
            ]
        )
    else:
        lines.append(f"- Label metrics unavailable: {label_evaluation.get('reason')}")

    lines.extend(["", "## Video Label Evaluation", ""])
    video_label_evaluation = report["video_label_evaluation"]
    if video_label_evaluation.get("available"):
        lines.extend(
            [
                "| Metric | Value |",
                "|---|---:|",
                f"| Labels | {video_label_evaluation['labels']} |",
                f"| Positive labels | {video_label_evaluation['positive_labels']} |",
                f"| Negative labels | {video_label_evaluation['negative_labels']} |",
                f"| Detected sources | {video_label_evaluation['detected_sources']} |",
                f"| Confirmed fall predictions | {video_label_evaluation['confirmed_positive_predictions']} |",
                f"| Human review positive predictions | {video_label_evaluation['review_positive_predictions']} |",
                f"| Rejected negative predictions | {video_label_evaluation['rejected_negative_predictions']} |",
                f"| Pending candidate sources | {video_label_evaluation['pending_sources']} |",
                f"| True positives | {video_label_evaluation['true_positives']} |",
                f"| False positives | {video_label_evaluation['false_positives']} |",
                f"| True negatives | {video_label_evaluation['true_negatives']} |",
                f"| False negatives | {video_label_evaluation['false_negatives']} |",
                f"| Accuracy | {video_label_evaluation['accuracy']} |",
                f"| Precision | {video_label_evaluation['precision']} |",
                f"| Recall | {video_label_evaluation['recall']} |",
                f"| F1 | {video_label_evaluation['f1']} |",
            ]
        )
    else:
        lines.append(
            f"- Video label metrics unavailable: {video_label_evaluation.get('reason')}"
        )

    lines.extend(["", "## Presentation Highlights", ""])
    highlights = report.get("presentation_highlights") or []
    if highlights:
        for highlight in highlights:
            lines.append(f"- {highlight}")
    else:
        lines.append("- No event output was found yet.")

    return "\n".join(lines) + "\n"


def write_metrics_report(
    report: Dict[str, Any],
    output_json: Path | str,
    output_markdown: Path | str,
) -> Dict[str, str]:
    output_json_path = Path(output_json)
    output_markdown_path = Path(output_markdown)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    output_markdown_path.write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )
    return {
        "json": str(output_json_path),
        "markdown": str(output_markdown_path),
    }


def default_output_paths(event_dir: Path | str) -> Dict[str, Path]:
    event_dir_path = Path(event_dir)
    return {
        "json": event_dir_path / REPORT_JSON_NAME,
        "markdown": event_dir_path / REPORT_MARKDOWN_NAME,
    }


def _average(values: Iterable[float], scale: float = 1.0) -> float:
    values = list(values)
    if not values:
        return 0.0
    return (sum(values) / len(values)) / scale


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "fall", "has_fall"}


def _source_matches(left: str, right: str) -> bool:
    left = str(left).strip()
    right = str(right).strip()
    if left == right:
        return True
    return Path(left).name == Path(right).name
