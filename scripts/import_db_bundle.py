from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.imports import build_import_plan


EXIT_VALID = 0
EXIT_INVALID_BUNDLE = 1
EXIT_UNSUPPORTED = 2
EXIT_DB_ERROR = 5
PostgresImportRepository = None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply a LexAI canonical DB import bundle.",
    )
    parser.add_argument("--source-dir", required=True, help="Canonical bundle output directory")
    parser.add_argument("--database-url", default=None, help="PostgreSQL URL; falls back to DATABASE_URL")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "apply"],
        default="dry_run",
    )
    parser.add_argument("--fail-on-orphan-edges", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--import-run-id", default=None)
    parser.add_argument("--with-embeddings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.with_embeddings:
        return _emit(
            {
                "ok": False,
                "status": "unsupported",
                "error": "with_embeddings is reserved for H08 Phase D3",
            },
            pretty=args.pretty,
            stream=sys.stderr,
            exit_code=EXIT_UNSUPPORTED,
        )

    database_url = (args.database_url or os.getenv("DATABASE_URL") or "").strip()
    if args.mode == "apply" and not database_url:
        return _emit(
            {
                "ok": False,
                "status": "unsupported",
                "error": "DATABASE_URL is required for apply mode",
            },
            pretty=args.pretty,
            stream=sys.stderr,
            exit_code=EXIT_UNSUPPORTED,
        )

    try:
        plan = build_import_plan(
            args.source_dir,
            with_embeddings=False,
            mode=args.mode,
            fail_on_orphan_edges=args.fail_on_orphan_edges,
            import_run_id=args.import_run_id,
        )
    except Exception:
        return _emit(
            {
                "ok": False,
                "status": "failed",
                "error": "import plan failed unexpectedly",
                "duration_seconds": _duration(started),
            },
            pretty=args.pretty,
            stream=sys.stderr,
            exit_code=EXIT_DB_ERROR,
        )

    if not plan.safe_for_db_import:
        return _emit(
            _summary(
                plan,
                ok=False,
                status="invalid_bundle",
                duration_seconds=_duration(started),
            ),
            pretty=args.pretty,
            exit_code=EXIT_INVALID_BUNDLE,
        )

    if args.mode == "dry_run":
        return _emit(
            _summary(
                plan,
                ok=True,
                status="dry_run",
                duration_seconds=_duration(started),
            ),
            pretty=args.pretty,
            exit_code=EXIT_VALID,
        )

    try:
        legal_units = _load_json_array(plan.artifact_paths.legal_units, "legal_units")
        legal_edges = _load_json_array(plan.artifact_paths.legal_edges, "legal_edges")
    except ValueError as exc:
        return _emit(
            {
                "ok": False,
                "status": "invalid_bundle",
                "import_run_id": plan.import_run_id,
                "source_dir": plan.source_dir,
                "errors": [{"code": "artifact_load_failed", "message": str(exc)}],
                "duration_seconds": _duration(started),
            },
            pretty=args.pretty,
            exit_code=EXIT_INVALID_BUNDLE,
        )

    try:
        repository_class = _get_repository_class()
        repository = repository_class(database_url)
        repository.ensure_schema()
        apply_result = repository.apply_import_plan(
            plan,
            legal_units=legal_units,
            legal_edges=legal_edges,
        )
    except Exception as exc:
        return _emit(
            {
                "ok": False,
                "status": "db_error",
                "import_run_id": plan.import_run_id,
                "source_dir": plan.source_dir,
                "error": "database import failed; transaction rolled back",
                "error_type": type(exc).__name__,
                "duration_seconds": _duration(started),
            },
            pretty=args.pretty,
            stream=sys.stderr,
            exit_code=EXIT_DB_ERROR,
        )

    return _emit(
        _summary(
            plan,
            ok=True,
            status=apply_result.status,
            duration_seconds=_duration(started),
            apply_counts=apply_result.counts,
        ),
        pretty=args.pretty,
        exit_code=EXIT_VALID,
    )


def _summary(
    plan,
    *,
    ok: bool,
    status: str,
    duration_seconds: float,
    apply_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = apply_counts or {
        "legal_units": {
            "attempted": plan.counts.legal_units,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "skipped": plan.counts.legal_units,
        },
        "legal_edges": {
            "attempted": plan.counts.legal_edges,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "skipped": plan.counts.legal_edges,
        },
        "embeddings": {
            "attempted": 0,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "skipped": 0,
        },
    }
    return {
        "ok": ok,
        "status": status,
        "import_run_id": plan.import_run_id,
        "source_dir": plan.source_dir,
        "mode": plan.mode,
        "safe_for_db_import": plan.safe_for_db_import,
        "table_names": ["legal_units", "legal_edges"],
        "counts": counts,
        "warnings": [warning.model_dump() for warning in plan.warnings],
        "errors": [error.model_dump() for error in plan.errors],
        "duration_seconds": duration_seconds,
    }


def _load_json_array(path: str | None, artifact: str) -> list[dict[str, Any]]:
    if path is None:
        raise ValueError(f"missing required artifact: {artifact}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact} is not valid JSON") from exc
    except OSError as exc:
        raise ValueError(f"{artifact} read failed") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{artifact} must contain a JSON array")
    records = [item for item in payload if isinstance(item, dict)]
    if len(records) != len(payload):
        raise ValueError(f"{artifact} must contain only JSON objects")
    return records


def _duration(started: float) -> float:
    return round(time.perf_counter() - started, 4)


def _get_repository_class():
    global PostgresImportRepository
    if PostgresImportRepository is None:
        from ingestion.import_repository import PostgresImportRepository as repository_class

        PostgresImportRepository = repository_class
    return PostgresImportRepository


def _emit(
    payload: dict[str, Any],
    *,
    pretty: bool,
    exit_code: int,
    stream=None,
) -> int:
    output_stream = stream or sys.stdout
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        ),
        file=output_stream,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
