from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.embeddings import (  # noqa: E402
    EmbeddingInputOutputValidationSummary,
    validate_embeddings_pair,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate embeddings_input.jsonl against embeddings_output.jsonl"
    )
    parser.add_argument("--input", required=True, help="Path to embeddings_input.jsonl")
    parser.add_argument("--output", required=True, help="Path to embeddings_output.jsonl")
    parser.add_argument("--expected-model", default=None)
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--require-all-inputs", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        summary = validate_embeddings_pair(
            Path(args.input),
            Path(args.output),
            expected_model=args.expected_model,
            expected_dim=args.expected_dim,
            require_all_inputs=args.require_all_inputs,
            strict=False,
        )
    except (OSError, ValueError) as exc:
        print(f"[validate-embeddings-pair] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(_format_human_report(summary))

    if args.strict and summary.errors:
        sys.exit(1)


def _format_human_report(summary: EmbeddingInputOutputValidationSummary) -> str:
    lines = [
        "Embeddings input/output pair validation",
        f"input_read_count: {summary.input_read_count}",
        f"output_read_count: {summary.output_read_count}",
        f"embeddable_input_count: {summary.embeddable_input_count}",
        f"matched_output_count: {summary.matched_output_count}",
        f"missing_output_count: {summary.missing_output_count}",
        f"orphan_output_count: {summary.orphan_output_count}",
        f"identity_mismatch_count: {summary.identity_mismatch_count}",
        f"duplicate_input_record_id_count: {summary.duplicate_input_record_id_count}",
        f"duplicate_output_resume_key_count: {summary.duplicate_output_resume_key_count}",
        f"unexpected_output_for_empty_input_count: {summary.unexpected_output_for_empty_input_count}",
        f"model_names: {_join_or_dash(summary.model_names)}",
        f"embedding_dims: {_join_or_dash([str(dim) for dim in summary.embedding_dims])}",
        f"law_ids: {_join_or_dash(summary.law_ids)}",
    ]
    _append_messages(lines, "warnings", summary.warnings)
    _append_messages(lines, "errors", summary.errors)
    return "\n".join(lines)


def _append_messages(lines: list[str], label: str, messages: list[str]) -> None:
    if not messages:
        return
    lines.append(f"{label}:")
    lines.extend(f"- {_truncate_message(message)}" for message in messages[:5])
    if len(messages) > 5:
        lines.append(f"- ... {len(messages) - 5} more")


def _join_or_dash(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def _truncate_message(message: str, limit: int = 240) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


if __name__ == "__main__":
    main()
