"""Evaluate CAUCAFall video-level fall detection results.

This script is intentionally narrower than generate_metrics_report.py. It focuses on
the demo evaluation flow: YOLO produces candidates, VLM assigns final statuses, and
CAUCAFall video labels provide TP/FP/TN/FN metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import EVENT_DIR
from services.metrics_report import load_event_metadata


DEFAULT_LABELS_PATH = Path("data") / "labels" / "caucafall_video_labels.csv"
DEFAULT_OUTPUT_JSON = Path("data") / "events" / "caucafall_eval_summary.json"
DEFAULT_OUTPUT_MD = Path("data") / "events" / "caucafall_eval_summary.md"
DEFAULT_OUTPUT_CSV = Path("data") / "events" / "caucafall_eval_details.csv"

POSITIVE_STATUSES = {"confirmed_fall", "need_human_review"}
NEGATIVE_STATUSES = {"rejected"}
PENDING_STATUSES = {"candidates"}

STATUS_LABELS = {
    "confirmed_fall": "确认摔倒",
    "need_human_review": "需要人工复核",
    "rejected": "已拒绝",
    "candidates": "候选待复核",
    "none": "未检出",
}

OUTCOME_LABELS = {
    "TP": "正确检出摔倒",
    "FP": "误报摔倒风险",
    "TN": "正确过滤正常视频",
    "FN": "漏检摔倒",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_caucafall_report(
        event_dir=Path(args.event_dir),
        labels_path=Path(args.labels_path),
    )
    paths = write_outputs(
        report=report,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        output_csv=Path(args.output_csv),
    )
    print(f"Wrote CAUCAFall JSON: {paths['json']}")
    print(f"Wrote CAUCAFall Markdown: {paths['markdown']}")
    print(f"Wrote CAUCAFall details CSV: {paths['details_csv']}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CAUCAFall video-level fall detection results."
    )
    parser.add_argument(
        "--event-dir",
        default=str(EVENT_DIR),
        help="Directory containing saved event metadata JSON files.",
    )
    parser.add_argument(
        "--labels-path",
        default=str(DEFAULT_LABELS_PATH),
        help="CSV with source_uri,has_fall columns.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Output Chinese Markdown report path.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output per-video detail CSV path.",
    )
    return parser


def build_caucafall_report(event_dir: Path | str, labels_path: Path | str) -> Dict[str, Any]:
    event_dir_path = Path(event_dir)
    labels_path_obj = Path(labels_path)
    labels = _load_video_labels(labels_path_obj)
    metadata = load_event_metadata(event_dir_path)
    events_by_source = _select_best_events_by_source(metadata)

    details: List[Dict[str, Any]] = []
    yolo_scores: List[float] = []
    vlm_confidences: List[float] = []

    vlm_counts = {
        "confirmed_fall": 0,
        "need_human_review": 0,
        "rejected": 0,
        "pending_candidates": 0,
    }
    metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
    }

    for label in labels:
        event = events_by_source.get(_source_key(label["source_uri"]))
        detail = _build_detail_row(label, event)
        details.append(detail)

        if event is not None:
            score = _safe_float(_candidate(event).get("score"))
            if score is not None:
                yolo_scores.append(score)
            confidence = _safe_float(_verification(event).get("confidence"))
            if confidence is not None:
                vlm_confidences.append(confidence)
            final_status = detail["final_status"]
            if final_status == "candidates":
                vlm_counts["pending_candidates"] += 1
            elif final_status in {"confirmed_fall", "need_human_review", "rejected"}:
                vlm_counts[final_status] += 1

        outcome = detail["outcome"]
        if outcome == "TP":
            metrics["true_positives"] += 1
        elif outcome == "FP":
            metrics["false_positives"] += 1
        elif outcome == "TN":
            metrics["true_negatives"] += 1
        elif outcome == "FN":
            metrics["false_negatives"] += 1

    total = len(labels)
    tp = metrics["true_positives"]
    fp = metrics["false_positives"]
    tn = metrics["true_negatives"]
    fn = metrics["false_negatives"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "event_dir": str(event_dir_path),
        "labels_path": str(labels_path_obj),
        "videos": {
            "total": total,
            "fall": sum(1 for label in labels if label["has_fall"]),
            "nofall": sum(1 for label in labels if not label["has_fall"]),
        },
        "yolo": {
            "candidates": len(yolo_scores),
            "average_score": _round(_average(yolo_scores)),
            "max_score": _round(max(yolo_scores) if yolo_scores else 0.0),
        },
        "vlm": {
            **vlm_counts,
            "average_confidence": _round(_average(vlm_confidences)),
        },
        "metrics": {
            **metrics,
            "accuracy": _round(_ratio(tp + tn, total)),
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        },
        "details": details,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    videos = report["videos"]
    yolo = report["yolo"]
    vlm = report["vlm"]
    metrics = report["metrics"]

    lines = [
        "# CAUCAFall 视频级评估报告",
        "",
        f"- 生成时间：`{report.get('generated_at', '')}`",
        f"- 事件目录：`{report['event_dir']}`",
        f"- 标签文件：`{report['labels_path']}`",
        "",
        "## 数据集概况",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 视频总数 | {videos['total']} |",
        f"| 摔倒视频 | {videos['fall']} |",
        f"| 正常视频 | {videos['nofall']} |",
        "",
        "## YOLO 与 VLM 汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| YOLO 候选数 | {yolo['candidates']} |",
        f"| YOLO 平均分 | {yolo['average_score']} |",
        f"| YOLO 最高分 | {yolo.get('max_score', 0.0)} |",
        f"| 确认摔倒 | {vlm['confirmed_fall']} |",
        f"| 需要人工复核 | {vlm['need_human_review']} |",
        f"| 已拒绝 | {vlm['rejected']} |",
        f"| 候选待复核 | {vlm['pending_candidates']} |",
        f"| VLM 平均置信度 | {vlm['average_confidence']} |",
        "",
        "## 视频级指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| TP 正确检出摔倒 | {metrics['true_positives']} |",
        f"| FP 误报摔倒风险 | {metrics['false_positives']} |",
        f"| TN 正确过滤正常视频 | {metrics['true_negatives']} |",
        f"| FN 漏检摔倒 | {metrics['false_negatives']} |",
        f"| Accuracy | {metrics['accuracy']} |",
        f"| Precision | {metrics['precision']} |",
        f"| Recall | {metrics['recall']} |",
        f"| F1 | {metrics['f1']} |",
        "",
        "## 视频明细",
        "",
        "| 视频 | 真实标签 | 系统判断 | 最终状态 | 结果 | YOLO 分数 | VLM 置信度 | VLM 理由 |",
        "|---|---|---|---|---|---:|---:|---|",
    ]

    for row in report["details"]:
        lines.append(
            "| {source_uri} | {true_label} | {prediction} | {final_status} | "
            "{outcome} | {yolo_score} | {vlm_confidence} | {vlm_reason} |".format(
                source_uri=_escape_md(row["source_uri"]),
                true_label="摔倒" if row["true_label"] == "fall" else "正常",
                prediction="摔倒风险" if row["prediction"] == "fall" else "未检出",
                final_status=_escape_md(
                    STATUS_LABELS.get(row["final_status"], row["final_status"])
                ),
                outcome=_escape_md(
                    f"{row['outcome']} {OUTCOME_LABELS.get(row['outcome'], '')}".strip()
                ),
                yolo_score=row["yolo_score"],
                vlm_confidence=row["vlm_confidence"],
                vlm_reason=_escape_md(row["vlm_reason"]),
            )
        )

    return "\n".join(lines) + "\n"


def write_outputs(
    report: Dict[str, Any],
    output_json: Path | str,
    output_md: Path | str,
    output_csv: Path | str,
) -> Dict[str, str]:
    output_json_path = Path(output_json)
    output_md_path = Path(output_md)
    output_csv_path = Path(output_csv)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    output_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_md_path.write_text(render_markdown(report), encoding="utf-8")
    _write_detail_csv(output_csv_path, report["details"])
    return {
        "json": str(output_json_path),
        "markdown": str(output_md_path),
        "details_csv": str(output_csv_path),
    }


def _load_video_labels(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels: List[Dict[str, Any]] = []
        for row in reader:
            source_uri = str(row.get("source_uri") or "").strip()
            if not source_uri:
                continue
            labels.append(
                {
                    "source_uri": source_uri,
                    "has_fall": _parse_bool(row.get("has_fall")),
                }
            )
    return labels


def _select_best_events_by_source(
    metadata: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for event in metadata:
        source_uri = str(event.get("source_uri") or _candidate(event).get("source_uri") or "")
        if not source_uri.strip():
            continue
        key = _source_key(source_uri)
        current = best.get(key)
        if current is None or _event_rank(event) > _event_rank(current):
            best[key] = event
    return best


def _build_detail_row(label: Dict[str, Any], event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    has_fall = bool(label["has_fall"])
    if event is None:
        final_status = "none"
        prediction = "nofall"
        candidate: Dict[str, Any] = {}
        verification: Dict[str, Any] = {}
    else:
        final_status = _final_status(event)
        prediction = "fall" if final_status in POSITIVE_STATUSES else "nofall"
        candidate = _candidate(event)
        verification = _verification(event)

    if has_fall and prediction == "fall":
        outcome = "TP"
    elif not has_fall and prediction == "fall":
        outcome = "FP"
    elif not has_fall and prediction == "nofall":
        outcome = "TN"
    else:
        outcome = "FN"

    return {
        "source_uri": label["source_uri"],
        "true_label": "fall" if has_fall else "nofall",
        "prediction": prediction,
        "final_status": final_status,
        "outcome": outcome,
        "yolo_score": _round(_safe_float(candidate.get("score")) or 0.0),
        "vlm_confidence": _round(_safe_float(verification.get("confidence")) or 0.0),
        "vlm_reason": str(
            verification.get("reason")
            or verification.get("failure_reason")
            or ("候选事件尚未完成 VLM 复核。" if final_status == "candidates" else "")
        ),
    }


def _write_detail_csv(path: Path, details: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "source_uri",
        "true_label",
        "prediction",
        "final_status",
        "outcome",
        "yolo_score",
        "vlm_confidence",
        "vlm_reason",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _candidate(event: Dict[str, Any]) -> Dict[str, Any]:
    candidate = event.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def _verification(event: Dict[str, Any]) -> Dict[str, Any]:
    verification = event.get("verification")
    return verification if isinstance(verification, dict) else {}


def _final_status(event: Dict[str, Any]) -> str:
    verification = _verification(event)
    result = str(verification.get("result") or "").strip()
    if result in POSITIVE_STATUSES or result in NEGATIVE_STATUSES:
        return result
    category = str(event.get("category") or "").strip()
    if category in POSITIVE_STATUSES or category in NEGATIVE_STATUSES or category in PENDING_STATUSES:
        return category
    return "none"


def _event_rank(event: Dict[str, Any]) -> int:
    final_status = _final_status(event)
    if final_status == "confirmed_fall":
        return 4
    if final_status == "need_human_review":
        return 3
    if final_status == "rejected":
        return 2
    if final_status == "candidates":
        return 1
    return 0


def _source_key(source_uri: str) -> str:
    return Path(str(source_uri).strip()).name.lower()


def _parse_bool(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "fall", "has_fall"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 3)


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
