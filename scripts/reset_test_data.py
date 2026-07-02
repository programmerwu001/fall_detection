"""Reset local test data for the fall gateway demo."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DB_PATH, DISABLED_DEBUG_EVENT_DIR, EVENT_DIR, PRIVATE_EVENT_DIR  # noqa: E402


def reset_test_data(
    event_dir: str | Path = EVENT_DIR,
    private_event_dir: str | Path | None = None,
    disabled_debug_event_dir: str | Path | None = None,
    db_path: str | Path = DB_PATH,
    workspace_root: str | Path = REPO_ROOT,
    dry_run: bool = True,
) -> dict:
    """Remove local event outputs and the SQLite database used by demos/tests."""
    root = Path(workspace_root).resolve()
    if private_event_dir is None:
        private_event_dir = Path(event_dir).parent / PRIVATE_EVENT_DIR.name
    if disabled_debug_event_dir is None:
        disabled_debug_event_dir = Path(event_dir).parent / DISABLED_DEBUG_EVENT_DIR.name
    events = _resolve_inside_workspace(event_dir, root, "event_dir")
    private_events = _resolve_inside_workspace(
        private_event_dir, root, "private_event_dir"
    )
    disabled_debug_events = _resolve_inside_workspace(
        disabled_debug_event_dir, root, "disabled_debug_event_dir"
    )
    database = _resolve_inside_workspace(db_path, root, "db_path")
    db_files = [path for path in _database_files(database) if path.exists()]
    event_entries = list(events.iterdir()) if events.exists() else []
    private_event_entries = list(private_events.iterdir()) if private_events.exists() else []
    disabled_debug_event_entries = (
        list(disabled_debug_events.iterdir()) if disabled_debug_events.exists() else []
    )

    if not dry_run:
        for path in db_files:
            path.unlink()
        _clear_directory_contents(events, event_entries)
        _clear_directory_contents(private_events, private_event_entries)
        _clear_directory_contents(
            disabled_debug_events, disabled_debug_event_entries
        )

    return {
        "mode": "dry-run" if dry_run else "deleted",
        "event_dir": str(events),
        "private_event_dir": str(private_events),
        "disabled_debug_event_dir": str(disabled_debug_events),
        "db_path": str(database),
        "db_files": [str(path) for path in db_files],
        "event_entries": len(event_entries),
        "private_event_entries": len(private_event_entries),
        "disabled_debug_event_entries": len(disabled_debug_event_entries),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clear local SQLite records and local event video outputs.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints targets.",
    )
    parser.add_argument("--event-dir", default=str(EVENT_DIR))
    parser.add_argument("--private-event-dir", default=str(PRIVATE_EVENT_DIR))
    parser.add_argument(
        "--disabled-debug-event-dir",
        default=str(DISABLED_DEBUG_EVENT_DIR),
    )
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--workspace-root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    result = reset_test_data(
        event_dir=args.event_dir,
        private_event_dir=args.private_event_dir,
        disabled_debug_event_dir=args.disabled_debug_event_dir,
        db_path=args.db_path,
        workspace_root=args.workspace_root,
        dry_run=not args.yes,
    )
    print(f"mode: {result['mode']}")
    print(f"event_dir: {result['event_dir']} ({result['event_entries']} entries)")
    print(
        "private_event_dir: "
        f"{result['private_event_dir']} ({result['private_event_entries']} entries)"
    )
    print(
        "disabled_debug_event_dir: "
        f"{result['disabled_debug_event_dir']} "
        f"({result['disabled_debug_event_entries']} entries)"
    )
    print(f"db_path: {result['db_path']}")
    print(f"db_files: {len(result['db_files'])}")
    for db_file in result["db_files"]:
        print(f"  {db_file}")
    if not args.yes:
        print("Run again with --yes to delete these files.")
    return 0


def _resolve_inside_workspace(path: str | Path, root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside workspace: {resolved}") from exc
    return resolved


def _database_files(db_path: Path) -> Iterable[Path]:
    yield db_path
    yield Path(str(db_path) + "-wal")
    yield Path(str(db_path) + "-shm")


def _clear_directory_contents(directory: Path, entries: Sequence[Path]) -> None:
    if directory.exists():
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
