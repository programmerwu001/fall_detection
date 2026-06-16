"""Generate a post-run metrics report for Fall Edge Gateway."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from config import DB_PATH, EVENT_DIR
from services.metrics_report import (
    build_metrics_report,
    default_output_paths,
    write_metrics_report,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    event_dir = Path(args.event_dir)
    defaults = default_output_paths(event_dir)
    output_json = Path(args.output_json) if args.output_json else defaults["json"]
    output_markdown = (
        Path(args.output_md) if args.output_md else defaults["markdown"]
    )
    queue_db_path = None if args.no_queue_db else Path(args.queue_db_path)

    report = build_metrics_report(
        event_dir=event_dir,
        queue_db_path=queue_db_path,
        labels_path=Path(args.labels_path) if args.labels_path else None,
        video_labels_path=(
            Path(args.video_labels_path) if args.video_labels_path else None
        ),
    )
    paths = write_metrics_report(
        report=report,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    print(f"Wrote metrics JSON: {paths['json']}")
    print(f"Wrote metrics Markdown: {paths['markdown']}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate JSON and Markdown metrics from saved event outputs."
    )
    parser.add_argument(
        "--event-dir",
        default=str(EVENT_DIR),
        help="Directory containing saved event metadata JSON files.",
    )
    parser.add_argument(
        "--queue-db-path",
        default=str(DB_PATH),
        help="SQLite queue database path. Ignored when --no-queue-db is set.",
    )
    parser.add_argument(
        "--no-queue-db",
        action="store_true",
        help="Do not read VLM queue metrics from SQLite.",
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Optional CSV labels file for Precision/Recall/F1/time accuracy.",
    )
    parser.add_argument(
        "--video-labels-path",
        default=None,
        help="Optional CSV video labels file with source_uri,has_fall columns.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path. Defaults to <event-dir>/metrics_summary.json.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown path. Defaults to <event-dir>/metrics_summary.md.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
