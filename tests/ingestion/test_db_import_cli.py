import json
from pathlib import Path

import scripts.import_db_bundle as import_cli


def test_cli_dry_run_does_not_require_database_url_or_write_db(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def fail_repository(*args, **kwargs):
        raise AssertionError("dry_run must not construct a DB repository")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--mode", "dry_run", "--pretty"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "dry_run"
    assert payload["counts"]["legal_units"]["attempted"] == 2
    assert "SECRET_RAW_TEXT" not in captured.out
    assert "SECRET_EMBEDDING_TEXT" not in captured.out
    assert "[0.1, 0.2]" not in captured.out


def test_cli_apply_without_database_url_returns_exit_code_2(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--mode", "apply"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert payload["error"] == "DATABASE_URL is required for apply mode"


def test_cli_apply_invalid_plan_returns_exit_code_1_without_repository(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path, missing={"legal_units.json"})

    def fail_repository(*args, **kwargs):
        raise AssertionError("invalid plans must not construct a DB repository")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
            "--pretty",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "invalid_bundle"
    assert any(error["code"] == "missing_required_artifact" for error in payload["errors"])


def test_cli_with_embeddings_is_refused_in_d2(tmp_path, capsys):
    source_dir = _write_bundle(tmp_path)

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--with-embeddings"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert payload["error"] == "with_embeddings is reserved for H08 Phase D3"


def test_cli_duplicate_ids_are_caught_before_db(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(
        tmp_path,
        legal_units=[_legal_unit("ro.test"), _legal_unit("ro.test")],
        legal_edges=[],
    )

    def fail_repository(*args, **kwargs):
        raise AssertionError("duplicate IDs must stop before DB repository construction")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", fail_repository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(error["code"] == "legal_units_duplicate_id" for error in payload["errors"])


def test_cli_orphan_edges_are_warning_by_default(tmp_path, capsys):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[_legal_edge("edge.orphan", "ro.test", "ro.test.missing")],
    )

    exit_code = import_cli.main(["--source-dir", str(source_dir), "--mode", "dry_run"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert any(warning["code"] == "orphan_legal_edges" for warning in payload["warnings"])


def test_cli_orphan_edges_are_error_when_configured(tmp_path, capsys):
    source_dir = _write_bundle(
        tmp_path,
        legal_edges=[_legal_edge("edge.orphan", "ro.test", "ro.test.missing")],
    )

    exit_code = import_cli.main(
        ["--source-dir", str(source_dir), "--mode", "dry_run", "--fail-on-orphan-edges"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert any(error["code"] == "orphan_legal_edges" for error in payload["errors"])


def test_cli_apply_success_uses_repository_and_sanitizes_summary(tmp_path, monkeypatch, capsys):
    source_dir = _write_bundle(tmp_path, with_embedding_artifacts=True)
    FakeRepository.instances = []
    monkeypatch.setattr(import_cli, "PostgresImportRepository", FakeRepository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "succeeded"
    assert payload["counts"]["legal_units"]["inserted"] == 2
    assert payload["counts"]["legal_edges"]["inserted"] == 1
    assert "SECRET_RAW_TEXT" not in captured.out
    assert "SECRET_EMBEDDING_TEXT" not in captured.out
    assert "[0.1, 0.2]" not in captured.out


def test_cli_apply_db_error_returns_exit_code_5_without_raw_text(
    tmp_path,
    monkeypatch,
    capsys,
):
    source_dir = _write_bundle(tmp_path)

    class FailingRepository(FakeRepository):
        def apply_import_plan(self, plan, *, legal_units, legal_edges):
            raise RuntimeError("SECRET_RAW_TEXT")

    monkeypatch.setattr(import_cli, "PostgresImportRepository", FailingRepository)

    exit_code = import_cli.main(
        [
            "--source-dir",
            str(source_dir),
            "--mode",
            "apply",
            "--database-url",
            "postgresql://lexai:test@localhost/lexai",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 5
    assert payload["status"] == "db_error"
    assert payload["error"] == "database import failed; transaction rolled back"
    assert "SECRET_RAW_TEXT" not in captured.err


class FakeApplyResult:
    status = "succeeded"

    @property
    def counts(self):
        return {
            "legal_units": {
                "attempted": 2,
                "inserted": 2,
                "updated": 0,
                "unchanged": 0,
                "skipped": 0,
            },
            "legal_edges": {
                "attempted": 1,
                "inserted": 1,
                "updated": 0,
                "unchanged": 0,
                "skipped": 0,
            },
            "embeddings": {
                "attempted": 0,
                "inserted": 0,
                "updated": 0,
                "unchanged": 0,
                "skipped": 0,
            },
        }


class FakeRepository:
    instances = []

    def __init__(self, database_url):
        self.database_url = database_url
        self.schema_ensured = False
        FakeRepository.instances.append(self)

    def ensure_schema(self):
        self.schema_ensured = True

    def apply_import_plan(self, plan, *, legal_units, legal_edges):
        assert self.schema_ensured is True
        assert len(list(legal_units)) == 2
        assert len(list(legal_edges)) == 1
        return FakeApplyResult()


def _write_bundle(
    tmp_path: Path,
    *,
    legal_units: list[dict] | None = None,
    legal_edges: list[dict] | None = None,
    missing: set[str] | None = None,
    with_embedding_artifacts: bool = False,
) -> Path:
    source_dir = tmp_path / "bundle"
    source_dir.mkdir()
    missing = missing or set()
    legal_units = legal_units or [_legal_unit("ro.test"), _legal_unit("ro.test.art_1")]
    legal_edges = legal_edges or [_legal_edge("edge.1", "ro.test", "ro.test.art_1")]
    payloads = {
        "legal_units.json": legal_units,
        "legal_edges.json": legal_edges,
        "reference_candidates.json": [],
        "legal_chunks.json": [],
        "validation_report.json": {"import_blocking_passed": True},
        "corpus_manifest.json": {"sources": []},
    }
    for filename, payload in payloads.items():
        if filename not in missing:
            _write_json(source_dir / filename, payload)
    if with_embedding_artifacts:
        (source_dir / "embeddings_input.jsonl").write_text(
            json.dumps(
                {
                    "record_id": "embedding.chunk.ro.test.art_1.0",
                    "chunk_id": "chunk.ro.test.art_1.0",
                    "legal_unit_id": "ro.test.art_1",
                    "law_id": "ro.test",
                    "text": "SECRET_EMBEDDING_TEXT",
                    "embedding_text": "SECRET_EMBEDDING_TEXT",
                    "text_hash": "hash-1",
                    "metadata": {"retrieval_text_is_citable": False},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (source_dir / "embeddings_output.jsonl").write_text(
            json.dumps(
                {
                    "record_id": "embedding.chunk.ro.test.art_1.0",
                    "chunk_id": "chunk.ro.test.art_1.0",
                    "legal_unit_id": "ro.test.art_1",
                    "law_id": "ro.test",
                    "model_name": "qwen3-embedding:4b",
                    "embedding_dim": 2,
                    "text_hash": "hash-1",
                    "embedding": [0.1, 0.2],
                    "metadata": {"retrieval_text_is_citable": False},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_json(
            source_dir / "validated_embeddings_manifest.json",
            {
                "ready_for_pgvector_import": True,
                "embedding_dim": 2,
                "law_ids": ["ro.test"],
            },
        )
    return source_dir


def _legal_unit(unit_id: str) -> dict:
    return {
        "id": unit_id,
        "canonical_id": unit_id,
        "source_id": "fixture_source",
        "law_id": "ro.test",
        "law_title": "Lege test",
        "status": "unknown",
        "hierarchy_path": ["Lege test"],
        "raw_text": "SECRET_RAW_TEXT",
        "normalized_text": "SECRET_RAW_TEXT",
        "legal_domain": "test",
        "legal_concepts": [],
        "parser_warnings": [],
    }


def _legal_edge(edge_id: str, source_id: str, target_id: str) -> dict:
    return {
        "id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "type": "contains",
        "weight": 1.0,
        "confidence": 1.0,
        "metadata": {"source": "unit-test"},
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
