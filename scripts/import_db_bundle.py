from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.imports import build_import_plan


EXIT_VALID = 0
EXIT_INVALID_BUNDLE = 1
EXIT_UNSUPPORTED = 2
EXIT_INVALID_EMBEDDINGS = 4
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
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--progress", action="store_true", help="Write safe import progress to stderr.")
    parser.add_argument(
        "--limit-units",
        type=_non_negative_int,
        default=None,
        help="Debug only: import only the first N legal_units records.",
    )
    parser.add_argument(
        "--limit-edges",
        type=_non_negative_int,
        default=None,
        help="Debug only: import only the first N legal_edges records.",
    )
    parser.add_argument(
        "--statement-timeout-seconds",
        type=_non_negative_int,
        default=0,
        help="Apply SET LOCAL statement_timeout inside the import transaction.",
    )
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Include sanitized exception details for DB errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    database_url = (
        args.database_url
        or os.getenv("DATABASE_ASYNCPG_URL")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()
    if args.mode == "apply" and not database_url:
        return _emit(
            {
                "ok": False,
                "status": "unsupported",
                "error": "DATABASE_URL or DATABASE_ASYNCPG_URL is required for apply mode",
            },
            pretty=args.pretty,
            stream=sys.stderr,
            exit_code=EXIT_UNSUPPORTED,
        )

    try:
        plan = build_import_plan(
            args.source_dir,
            with_embeddings=args.with_embeddings,
            mode=args.mode,
            embedding_dim=args.embedding_dim,
            fail_on_orphan_edges=args.fail_on_orphan_edges,
            import_run_id=args.import_run_id,
        )
    except ValueError as exc:
        return _emit(
            {
                "ok": False,
                "status": "unsupported",
                "error": str(exc),
                "duration_seconds": _duration(started),
            },
            pretty=args.pretty,
            stream=sys.stderr,
            exit_code=EXIT_UNSUPPORTED,
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
                with_embeddings=args.with_embeddings,
                embedding_dim=args.embedding_dim if args.with_embeddings else None,
                **_summary_options(args),
            ),
            pretty=args.pretty,
            exit_code=_invalid_plan_exit_code(args, plan),
        )

    if args.mode == "dry_run":
        return _emit(
            _summary(
                plan,
                ok=True,
                status="dry_run",
                duration_seconds=_duration(started),
                with_embeddings=args.with_embeddings,
                embedding_dim=args.embedding_dim if args.with_embeddings else None,
                **_summary_options(args),
            ),
            pretty=args.pretty,
            exit_code=EXIT_VALID,
        )

    try:
        legal_units = _load_json_array(plan.artifact_paths.legal_units, "legal_units")
        legal_edges = _load_json_array(plan.artifact_paths.legal_edges, "legal_edges")
        embeddings = (
            _load_embedding_records(
                plan.artifact_paths.embeddings_output,
                plan.artifact_paths.embeddings_input,
                expected_dim=args.embedding_dim,
            )
            if args.with_embeddings
            else None
        )
        legal_units = _limit_records(legal_units, args.limit_units)
        legal_edges = _limit_records(legal_edges, args.limit_edges)
    except ValueError as exc:
        return _emit(
            {
                "ok": False,
                "status": "invalid_bundle",
                "import_run_id": plan.import_run_id,
                "source_dir": plan.source_dir,
                "errors": [{"code": "artifact_load_failed", "message": str(exc)}],
                "duration_seconds": _duration(started),
                **_summary_options(args),
            },
            pretty=args.pretty,
            exit_code=EXIT_INVALID_EMBEDDINGS if args.with_embeddings else EXIT_INVALID_BUNDLE,
        )

    try:
        repository_class = _get_repository_class()
        repository = repository_class(database_url)
        repository.ensure_schema()
        progress_callback = _progress_callback(args.progress)
        apply_kwargs: dict[str, Any] = {
            "plan": plan,
            "legal_units": legal_units,
            "legal_edges": legal_edges,
        }
        if progress_callback is not None:
            apply_kwargs["progress_callback"] = progress_callback
        if args.statement_timeout_seconds:
            apply_kwargs["statement_timeout_seconds"] = args.statement_timeout_seconds
        if args.with_embeddings:
            apply_kwargs["embeddings"] = embeddings
            apply_kwargs["expected_embedding_dim"] = args.embedding_dim
            apply_result = repository.apply_import_plan(
                **apply_kwargs,
            )
        else:
            apply_result = repository.apply_import_plan(
                **apply_kwargs,
            )
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "db_error",
            "import_run_id": plan.import_run_id,
            "source_dir": plan.source_dir,
            "error": "database import failed; transaction rolled back",
            "error_type": type(exc).__name__,
            "duration_seconds": _duration(started),
            **_summary_options(args),
        }
        if args.debug_errors:
            payload.update(_debug_error_payload(exc))
        return _emit(
            payload,
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
            with_embeddings=args.with_embeddings,
            embedding_dim=args.embedding_dim if args.with_embeddings else None,
            **_summary_options(args),
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
    with_embeddings: bool = False,
    embedding_dim: int | None = None,
    limited_debug_run: bool = False,
    limit_units: int | None = None,
    limit_edges: int | None = None,
    statement_timeout_seconds: int = 0,
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
            "attempted": plan.counts.embedding_records if with_embeddings else 0,
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
            "skipped": plan.counts.embedding_records if with_embeddings else 0,
        },
    }
    counts = _summary_counts(counts)
    model_name = _embedding_model_name(plan.artifact_paths.embeddings_manifest) if with_embeddings else None
    return {
        "ok": ok,
        "status": status,
        "import_run_id": plan.import_run_id,
        "source_dir": plan.source_dir,
        "mode": plan.mode,
        "safe_for_db_import": plan.safe_for_db_import,
        "table_names": (
            ["legal_units", "legal_edges", "legal_embeddings"]
            if with_embeddings
            else ["legal_units", "legal_edges"]
        ),
        "counts": counts,
        "embedding_dim": embedding_dim,
        "model_name": model_name,
        "limited_debug_run": limited_debug_run,
        "limits": {
            "legal_units": limit_units,
            "legal_edges": limit_edges,
        },
        "statement_timeout_seconds": statement_timeout_seconds,
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


def _load_embedding_records(
    output_path: str | None,
    input_path: str | None,
    *,
    expected_dim: int,
) -> list[dict[str, Any]]:
    from ingestion.embeddings import validate_embedding_vector

    if output_path is None:
        raise ValueError("missing required artifact: embeddings_output")
    input_mappings = _load_embedding_input_mappings(input_path)
    records = _load_jsonl_objects(output_path, "embeddings_output")
    for index, record in enumerate(records, start=1):
        record_id = record.get("record_id")
        metadata = record.get("metadata")
        if metadata is None:
            metadata = {}
            record["metadata"] = metadata
        if not isinstance(metadata, dict):
            raise ValueError(f"embeddings_output line {index} metadata must be a JSON object")
        mapping = input_mappings.get(record_id) if isinstance(record_id, str) else None
        if not record.get("legal_unit_id") and mapping:
            record["legal_unit_id"] = mapping.get("legal_unit_id")
        if not record.get("chunk_id") and mapping:
            record["chunk_id"] = mapping.get("chunk_id")
        if not record.get("legal_unit_id"):
            record["legal_unit_id"] = metadata.get("legal_unit_id")
        if not record.get("chunk_id"):
            record["chunk_id"] = metadata.get("chunk_id")
        if not record.get("legal_unit_id"):
            raise ValueError(f"embeddings_output line {index} missing legal_unit_id mapping")
        for field in ("record_id", "model_name", "text_hash"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"embeddings_output line {index} missing {field}")
        embedding_dim = record.get("embedding_dim")
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise ValueError(f"embeddings_output line {index} missing embedding_dim")
        if embedding_dim != expected_dim:
            raise ValueError(
                f"embeddings_output line {index} dimension mismatch: expected {expected_dim}, got {embedding_dim}"
            )
        try:
            validate_embedding_vector(record.get("embedding"), expected_dim=expected_dim)
        except ValueError as exc:
            raise ValueError(f"embeddings_output line {index} invalid embedding vector") from exc
        record["source_path"] = str(Path(output_path))
    return records


def _load_embedding_input_mappings(input_path: str | None) -> dict[str, dict[str, str | None]]:
    if input_path is None:
        return {}
    mappings: dict[str, dict[str, str | None]] = {}
    for index, record in enumerate(_load_jsonl_objects(input_path, "embeddings_input"), start=1):
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"embeddings_input line {index} missing record_id")
        mappings[record_id] = {
            "legal_unit_id": _optional_string(record.get("legal_unit_id")),
            "chunk_id": _optional_string(record.get("chunk_id")),
        }
    return mappings


def _load_jsonl_objects(path: str | Path, artifact: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{artifact} read failed") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{artifact} line {line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{artifact} line {line_number} must contain a JSON object")
        records.append(value)
    return records


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def _summary_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "limited_debug_run": args.limit_units is not None or args.limit_edges is not None,
        "limit_units": args.limit_units,
        "limit_edges": args.limit_edges,
        "statement_timeout_seconds": args.statement_timeout_seconds,
    }


def _progress_callback(enabled: bool):
    if not enabled:
        return None

    def emit(message: str) -> None:
        print(_sanitize_debug_text(message), file=sys.stderr, flush=True)

    return emit


def _invalid_plan_exit_code(args: argparse.Namespace, plan) -> int:
    if args.mode == "apply" and args.with_embeddings and _has_embedding_errors(plan):
        return EXIT_INVALID_EMBEDDINGS
    return EXIT_INVALID_BUNDLE


def _has_embedding_errors(plan) -> bool:
    return any(
        error.code.startswith("embedding")
        or error.code.startswith("embeddings")
        or error.code == "missing_required_embedding_artifact"
        or error.artifact in {"embeddings_input", "embeddings_output", "embeddings_manifest"}
        for error in plan.errors
    )


def _summary_counts(counts: dict[str, Any]) -> dict[str, Any]:
    return {name: _count_payload(values) for name, values in counts.items()}


def _count_payload(values: Any) -> dict[str, int]:
    payload = dict(values)
    for field in ("attempted", "inserted", "updated", "unchanged", "failed", "skipped"):
        payload.setdefault(field, 0)
    payload["attempted_count"] = payload["attempted"]
    payload["inserted_count"] = payload["inserted"]
    payload["updated_count"] = payload["updated"]
    payload["unchanged_count"] = payload["unchanged"]
    payload["failed_count"] = payload["failed"]
    return payload


def _embedding_model_name(path: str | None) -> str | None:
    if path is None:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    model_name = payload.get("model_name")
    return model_name if isinstance(model_name, str) and model_name.strip() else None


def _duration(started: float) -> float:
    return round(time.perf_counter() - started, 4)


def _debug_error_payload(exc: Exception) -> dict[str, Any]:
    traceback_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "error_message": _sanitize_debug_text(str(exc)),
        "error_repr": _sanitize_debug_text(repr(exc)),
        "traceback_tail": [
            _sanitize_debug_text(line.rstrip("\n")) for line in traceback_lines[-20:]
        ],
    }


def _sanitize_debug_text(text: str) -> str:
    sanitized = text
    for pattern, replacement in _DEBUG_REDACTIONS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


_DEBUG_REDACTIONS = (
    (
        re.compile(r"(?i)\b(postgres(?:ql)?(?:\+asyncpg)?://)([^:@/\s]+):([^@/\s]+)@"),
        r"\1\2:***@",
    ),
    (re.compile(r"(?i)(\bpassword=)([^&\s;]+)"), r"\1***"),
    (re.compile(r"(?i)(\b(?:DATABASE_URL|DATABASE_ASYNCPG_URL)=)(\S+)"), r"\1***"),
    (
        re.compile(
            r"(?is)((?:['\"])?(?:raw_text|normalized_text|embedding_text)(?:['\"])?\s*[:=]\s*)(['\"])(.*?)(\2)"
        ),
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>{match.group(4)}",
    ),
    (
        re.compile(r"(?is)((?:['\"])?embedding(?:['\"])?\s*[:=]\s*)\[[^\]]*\]"),
        r"\1[<redacted>]",
    ),
)


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
