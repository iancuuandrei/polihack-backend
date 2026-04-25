import json
import sys

import pytest

import scripts.validate_embeddings_pair as pair_cli
from ingestion.embeddings import validate_embeddings_pair


MODEL = "Qwen/Qwen3-Embedding-4B"


def test_valid_input_and_output_returns_summary(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1", "2"], ["1", "2"])

    summary = validate_embeddings_pair(
        input_path,
        output_path,
        expected_model=MODEL,
        expected_dim=3,
    )

    assert summary.input_read_count == 2
    assert summary.output_read_count == 2
    assert summary.embeddable_input_count == 2
    assert summary.matched_output_count == 2
    assert summary.missing_output_count == 0
    assert summary.orphan_output_count == 0
    assert summary.identity_mismatch_count == 0
    assert summary.model_names == [MODEL]
    assert summary.embedding_dims == [3]
    assert summary.law_ids == ["ro.test"]
    assert summary.errors == []


def test_orphan_output_record_id_is_error(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["2"])

    summary = validate_embeddings_pair(input_path, output_path, strict=False)

    assert summary.orphan_output_count == 1
    assert any("orphan output record_id" in error for error in summary.errors)


@pytest.mark.parametrize("field", ["chunk_id", "legal_unit_id", "law_id", "text_hash"])
def test_identity_mismatch_fields_are_errors(tmp_path, field):
    input_record = _input_record("1")
    output_record = _output_record_from_input(input_record)
    output_record[field] = f"changed-{field}"
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])

    summary = validate_embeddings_pair(input_path, output_path, strict=False)

    assert summary.identity_mismatch_count == 1
    assert any(f"identity mismatch fields={field}" in error for error in summary.errors)


def test_missing_output_for_embeddable_input_is_error_when_required(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1", "2"], ["1"])

    summary = validate_embeddings_pair(
        input_path,
        output_path,
        require_all_inputs=True,
        strict=False,
    )

    assert summary.missing_output_count == 1
    assert any("missing embedding output" in error for error in summary.errors)


def test_missing_output_for_embeddable_input_can_be_warning(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1", "2"], ["1"])

    summary = validate_embeddings_pair(
        input_path,
        output_path,
        require_all_inputs=False,
        strict=False,
    )

    assert summary.missing_output_count == 1
    assert summary.errors == []
    assert any("missing embedding output" in warning for warning in summary.warnings)


def test_empty_embedding_input_does_not_require_output(tmp_path):
    input_record = _input_record("1", embedding_text="")
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [])

    summary = validate_embeddings_pair(input_path, output_path)

    assert summary.embeddable_input_count == 0
    assert summary.missing_output_count == 0
    assert summary.errors == []


def test_output_for_empty_embedding_input_is_error(tmp_path):
    input_record = _input_record("1", embedding_text="")
    output_record = _output_record_from_input(input_record)
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])

    summary = validate_embeddings_pair(input_path, output_path, strict=False)

    assert summary.unexpected_output_for_empty_input_count == 1
    assert any("unexpected output for non-embeddable input" in error for error in summary.errors)


def test_duplicate_input_record_id_is_error(tmp_path):
    records = [_input_record("1"), _input_record("1", text_hash="hash-b")]
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, records)
    _write_jsonl(output_path, [_output_record_from_input(records[0])])

    summary = validate_embeddings_pair(input_path, output_path, strict=False)

    assert summary.duplicate_input_record_id_count == 1
    assert any("duplicate input record_id" in error for error in summary.errors)


def test_duplicate_output_resume_key_is_error(tmp_path):
    input_record = _input_record("1")
    output_record = _output_record_from_input(input_record)
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record, dict(output_record)])

    summary = validate_embeddings_pair(input_path, output_path, strict=False)

    assert summary.duplicate_output_resume_key_count == 1
    assert any("duplicate resume key" in error for error in summary.errors)


def test_expected_model_mismatch_is_error(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["1"])

    summary = validate_embeddings_pair(
        input_path,
        output_path,
        expected_model="other-model",
        strict=False,
    )

    assert any("expected_model mismatch" in error for error in summary.errors)


def test_expected_dim_mismatch_is_error(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["1"])

    summary = validate_embeddings_pair(
        input_path,
        output_path,
        expected_dim=2560,
        strict=False,
    )

    assert any("expected_dim mismatch" in error for error in summary.errors)


def test_strict_false_returns_errors_without_exception(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["2"])

    summary = validate_embeddings_pair(input_path, output_path, strict=False)

    assert summary.errors


def test_strict_true_raises_value_error(tmp_path):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["2"])

    with pytest.raises(ValueError, match="embedding pair validation failed"):
        validate_embeddings_pair(input_path, output_path, strict=True)


def test_cli_human_readable_passes_on_valid_pair(tmp_path, monkeypatch, capsys):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["1"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_pair.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--expected-model",
            MODEL,
            "--expected-dim",
            "3",
            "--require-all-inputs",
            "--strict",
        ],
    )

    pair_cli.main()

    stdout = capsys.readouterr().out
    assert "Embeddings input/output pair validation" in stdout
    assert "matched_output_count: 1" in stdout
    assert "missing_output_count: 0" in stdout


def test_cli_json_outputs_valid_json(tmp_path, monkeypatch, capsys):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["1"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_pair.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--json",
        ],
    )

    pair_cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["input_read_count"] == 1
    assert payload["matched_output_count"] == 1
    assert payload["model_names"] == [MODEL]


def test_cli_strict_returns_nonzero_on_invalid_pair(tmp_path, monkeypatch):
    input_path, output_path = _write_pair(tmp_path, ["1"], ["2"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_pair.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        pair_cli.main()

    assert exc.value.code == 1


def test_validator_does_not_require_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    input_path, output_path = _write_pair(tmp_path, ["1"], ["1"])

    summary = validate_embeddings_pair(input_path, output_path)

    assert summary.matched_output_count == 1


def test_cli_output_does_not_include_text_fields_or_full_vector(tmp_path, monkeypatch, capsys):
    secret_text = "SECRET_FULL_LEGAL_TEXT"
    input_record = _input_record("1", text=secret_text, embedding_text="SECRET_EMBEDDING_TEXT")
    output_record = _output_record_from_input(input_record, embedding=[0.1, 0.2, 0.3], embedding_dim=2)
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_embeddings_pair.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    pair_cli.main()

    stdout = capsys.readouterr().out
    assert "SECRET_FULL_LEGAL_TEXT" not in stdout
    assert "SECRET_EMBEDDING_TEXT" not in stdout
    assert "[0.1, 0.2, 0.3]" not in stdout


def test_strict_exception_does_not_include_text_fields_or_full_vector(tmp_path):
    input_record = _input_record("1", text="SECRET_FULL_LEGAL_TEXT", embedding_text="SECRET_EMBEDDING_TEXT")
    output_record = _output_record_from_input(input_record, embedding=[0.1, 0.2, 0.3], embedding_dim=2)
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, [input_record])
    _write_jsonl(output_path, [output_record])

    with pytest.raises(ValueError) as exc:
        validate_embeddings_pair(input_path, output_path)

    message = str(exc.value)
    assert "SECRET_FULL_LEGAL_TEXT" not in message
    assert "SECRET_EMBEDDING_TEXT" not in message
    assert "[0.1, 0.2, 0.3]" not in message


def _write_pair(tmp_path, input_suffixes: list[str], output_suffixes: list[str]):
    inputs = [_input_record(suffix) for suffix in input_suffixes]
    inputs_by_suffix = {
        record["record_id"].split(".")[-2]: record
        for record in inputs
    }
    outputs = [
        _output_record_from_input(inputs_by_suffix[suffix])
        if suffix in inputs_by_suffix
        else _output_record(suffix)
        for suffix in output_suffixes
    ]
    input_path = tmp_path / "embeddings_input.jsonl"
    output_path = tmp_path / "embeddings_output.jsonl"
    _write_jsonl(input_path, inputs)
    _write_jsonl(output_path, outputs)
    return input_path, output_path


def _input_record(
    suffix: str,
    *,
    text: str = "LegalUnit.raw_text",
    embedding_text: str = "LegalChunk.embedding_text",
    text_hash: str | None = None,
) -> dict:
    if text_hash is None:
        text_hash = f"hash-{suffix}"
    return {
        "record_id": f"embedding.chunk.ro.test.{suffix}.0",
        "chunk_id": f"chunk.ro.test.{suffix}.0",
        "legal_unit_id": f"ro.test.{suffix}",
        "law_id": "ro.test",
        "text": text,
        "embedding_text": embedding_text,
        "text_hash": text_hash,
        "model_hint": None,
        "metadata": {"retrieval_text_is_citable": False},
    }


def _output_record_from_input(
    input_record: dict,
    *,
    model_name: str = MODEL,
    embedding: list | None = None,
    embedding_dim: int = 3,
) -> dict:
    return {
        "record_id": input_record["record_id"],
        "chunk_id": input_record["chunk_id"],
        "legal_unit_id": input_record["legal_unit_id"],
        "law_id": input_record["law_id"],
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "text_hash": input_record["text_hash"],
        "embedding": embedding if embedding is not None else [0.1, 0.2, 0.3],
        "metadata": {"retrieval_text_is_citable": False},
    }


def _output_record(suffix: str, *, model_name: str = MODEL) -> dict:
    return {
        "record_id": f"embedding.chunk.ro.test.{suffix}.0",
        "chunk_id": f"chunk.ro.test.{suffix}.0",
        "legal_unit_id": f"ro.test.{suffix}",
        "law_id": "ro.test",
        "model_name": model_name,
        "embedding_dim": 3,
        "text_hash": f"hash-{suffix}",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"retrieval_text_is_citable": False},
    }


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
