from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.pipeline import run_pipeline


DEFAULT_SOURCES_PATH = REPO_ROOT / "config" / "legal_sources.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ingestion" / "output"
VALID_STATUSES = {"active", "historical", "repealed", "unknown"}
REQUIRED_SOURCE_FIELDS = ("source_id", "law_id", "law_title", "source_url")


def load_legal_sources(sources_path: str | Path) -> list[dict[str, Any]]:
    path = Path(sources_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"sources file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"sources file is not valid JSON: {path}: {exc.msg}") from exc

    if not isinstance(data, list):
        raise ValueError("sources file must contain a JSON array")

    sources: list[dict[str, Any]] = []
    for index, raw_source in enumerate(data):
        if not isinstance(raw_source, dict):
            raise ValueError(f"source[{index}] must be a JSON object")
        sources.append(_validate_source(raw_source, index=index))
    return sources


def run_daily_ingestion(
    *,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    limit_sources: int | None = None,
    dry_run: bool = False,
    write_debug: bool = False,
) -> dict[str, Any]:
    if limit_sources is not None and limit_sources < 1:
        raise ValueError("limit_sources must be a positive integer")

    sources = load_legal_sources(sources_path)
    enabled_sources = [source for source in sources if source.get("enabled", True)]
    selected_sources = enabled_sources[:limit_sources] if limit_sources else enabled_sources
    root = Path(output_root)
    output_dirs = [
        str(_output_dir_for_source(source, root)) for source in selected_sources
    ]

    summary: dict[str, Any] = {
        "sources_total": len(sources),
        "sources_enabled": len(enabled_sources),
        "sources_succeeded": 0,
        "sources_failed": 0,
        "output_dirs": output_dirs,
        "errors": [],
    }

    if dry_run:
        return summary

    for source in selected_sources:
        out_dir = _output_dir_for_source(source, root)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            run_pipeline(
                url=source["source_url"],
                out_dir=out_dir,
                law_id=source["law_id"],
                law_title=source["law_title"],
                source_id=source["source_id"],
                status=source.get("status", "unknown"),
                write_debug=write_debug,
            )
        except Exception as exc:
            summary["sources_failed"] += 1
            summary["errors"].append(
                {
                    "source_id": source["source_id"],
                    "law_id": source["law_id"],
                    "error": str(exc),
                }
            )
        else:
            summary["sources_succeeded"] += 1

    return summary


def sanitize_law_id(law_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", law_id.strip()).strip("_")
    if not sanitized:
        raise ValueError("law_id cannot be sanitized to an output directory name")
    return sanitized


def _validate_source(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    validated = dict(source)
    for field in REQUIRED_SOURCE_FIELDS:
        value = validated.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source[{index}].{field} must be a non-empty string")
        validated[field] = value.strip()

    status = validated.get("status", "unknown")
    if not isinstance(status, str) or status not in VALID_STATUSES:
        raise ValueError(
            f"source[{index}].status must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )
    validated["status"] = status

    enabled = validated.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"source[{index}].enabled must be a boolean")
    validated["enabled"] = enabled

    parsed_url = urlparse(validated["source_url"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError(f"source[{index}].source_url must be an http(s) URL")

    sanitize_law_id(validated["law_id"])
    return validated


def _output_dir_for_source(source: dict[str, Any], output_root: Path) -> Path:
    return output_root / sanitize_law_id(source["law_id"])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily LexAI canonical ingestion for explicit legal sources."
    )
    parser.add_argument(
        "--sources",
        default=str(DEFAULT_SOURCES_PATH),
        help="Path to legal sources JSON file",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for canonical parser bundles",
    )
    parser.add_argument(
        "--limit-sources",
        type=int,
        default=None,
        help="Process at most this many enabled sources",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source config and print planned output dirs without scraping",
    )
    parser.add_argument(
        "--write-debug",
        action="store_true",
        help="Write parser debug files next to each canonical bundle",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summary = run_daily_ingestion(
            sources_path=args.sources,
            output_root=args.output_root,
            limit_sources=args.limit_sources,
            dry_run=args.dry_run,
            write_debug=args.write_debug,
        )
    except ValueError as exc:
        summary = {
            "sources_total": 0,
            "sources_enabled": 0,
            "sources_succeeded": 0,
            "sources_failed": 1,
            "output_dirs": [],
            "errors": [{"error": str(exc)}],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        sys.exit(2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    sys.exit(1 if summary["sources_failed"] else 0)


if __name__ == "__main__":
    main()
