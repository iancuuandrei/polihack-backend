import copy
import json
import os
from pathlib import Path

import pytest

from ingestion.import_repository import PostgresImportRepository
from ingestion.imports import build_import_plan


def test_postgres_import_repository_upsert_legal_units_is_idempotent():
    connection = FakeConnection()
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )
    record = _legal_unit("ro.test.art_1", raw_text="SECRET_RAW_TEXT")

    first = repository.upsert_legal_units([record])
    second = repository.upsert_legal_units([record])

    assert first.attempted == 1
    assert first.inserted == 1
    assert first.updated == 0
    assert first.unchanged == 0
    assert second.attempted == 1
    assert second.inserted == 0
    assert second.updated == 0
    assert second.unchanged == 1
    assert len(connection.legal_units) == 1


def test_postgres_import_repository_upsert_legal_units_updates_changed_rows():
    connection = FakeConnection()
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )
    repository.upsert_legal_units([_legal_unit("ro.test.art_1", raw_text="first")])

    result = repository.upsert_legal_units([_legal_unit("ro.test.art_1", raw_text="second")])

    assert result.inserted == 0
    assert result.updated == 1
    assert result.unchanged == 0
    assert len(connection.legal_units) == 1


def test_postgres_import_repository_upsert_legal_edges_is_idempotent():
    connection = FakeConnection()
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )
    record = _legal_edge("edge.1", "ro.test", "ro.test.art_1")

    first = repository.upsert_legal_edges([record])
    second = repository.upsert_legal_edges([record])

    assert first.attempted == 1
    assert first.inserted == 1
    assert second.attempted == 1
    assert second.unchanged == 1
    assert len(connection.legal_edges) == 1


def test_postgres_import_repository_upsert_embeddings_is_idempotent():
    connection = FakeConnection()
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )
    record = _embedding_record()

    first = repository.upsert_embeddings([record], expected_dim=2)
    second = repository.upsert_embeddings([record], expected_dim=2)

    assert first.attempted == 1
    assert first.inserted == 1
    assert first.updated == 0
    assert second.attempted == 1
    assert second.unchanged == 1
    assert second.failed == 0
    assert len(connection.legal_embeddings) == 1


def test_postgres_import_repository_upsert_embeddings_updates_changed_rows():
    connection = FakeConnection()
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )
    repository.upsert_embeddings([_embedding_record(embedding=[0.1, 0.2])], expected_dim=2)

    result = repository.upsert_embeddings(
        [_embedding_record(embedding=[0.3, 0.4])],
        expected_dim=2,
    )

    assert result.inserted == 0
    assert result.updated == 1
    assert result.unchanged == 0
    assert len(connection.legal_embeddings) == 1


def test_postgres_import_repository_upsert_embeddings_validates_expected_dim():
    repository = PostgresImportRepository("postgresql://lexai:test@localhost/lexai")

    with pytest.raises(ValueError, match="dimension mismatch"):
        repository.upsert_embeddings([_embedding_record(embedding=[0.1, 0.2])], expected_dim=3)


def test_apply_import_plan_rolls_back_units_when_edges_fail(tmp_path):
    source_dir = _write_bundle(tmp_path)
    plan = build_import_plan(source_dir, mode="apply")
    connection = FakeConnection(fail_on_edge=True)
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )

    with pytest.raises(RuntimeError, match="edge upsert failed"):
        repository.apply_import_plan(
            plan,
            legal_units=[_legal_unit("ro.test.art_1")],
            legal_edges=[_legal_edge("edge.1", "ro.test", "ro.test.art_1")],
        )

    assert connection.legal_units == {}
    assert connection.legal_edges == {}
    assert connection.import_runs == {}
    assert connection.rolled_back is True


def test_apply_import_plan_rolls_back_units_and_edges_when_embeddings_fail(tmp_path):
    source_dir = _write_bundle(tmp_path, with_embeddings=True)
    plan = build_import_plan(source_dir, mode="apply", with_embeddings=True, embedding_dim=2)
    connection = FakeConnection(fail_on_embedding=True)
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )

    with pytest.raises(RuntimeError, match="embedding upsert failed"):
        repository.apply_import_plan(
            plan,
            legal_units=[_legal_unit("ro.test"), _legal_unit("ro.test.art_1")],
            legal_edges=[_legal_edge("edge.1", "ro.test", "ro.test.art_1")],
            embeddings=[_embedding_record()],
            expected_embedding_dim=2,
        )

    assert connection.legal_units == {}
    assert connection.legal_edges == {}
    assert connection.legal_embeddings == {}
    assert connection.import_runs == {}
    assert connection.rolled_back is True


def test_repeated_apply_does_not_duplicate_embeddings(tmp_path):
    source_dir = _write_bundle(tmp_path, with_embeddings=True)
    plan = build_import_plan(source_dir, mode="apply", with_embeddings=True, embedding_dim=2)
    connection = FakeConnection()
    repository = PostgresImportRepository(
        "postgresql://lexai:test@localhost/lexai",
        connect_func=lambda _: connection,
    )

    first = repository.apply_import_plan(
        plan,
        legal_units=[_legal_unit("ro.test"), _legal_unit("ro.test.art_1")],
        legal_edges=[_legal_edge("edge.1", "ro.test", "ro.test.art_1")],
        embeddings=[_embedding_record()],
        expected_embedding_dim=2,
    )
    second = repository.apply_import_plan(
        plan,
        legal_units=[_legal_unit("ro.test"), _legal_unit("ro.test.art_1")],
        legal_edges=[_legal_edge("edge.1", "ro.test", "ro.test.art_1")],
        embeddings=[_embedding_record()],
        expected_embedding_dim=2,
    )

    assert first.embeddings.inserted == 1
    assert second.embeddings.unchanged == 1
    assert len(connection.legal_embeddings) == 1


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL_TEST"),
    reason="DATABASE_URL_TEST is not configured for PostgreSQL integration tests",
)
def test_postgres_import_repository_integration_idempotent_unit_upsert():
    database_url = os.environ["DATABASE_URL_TEST"]
    repository = PostgresImportRepository(database_url)
    repository.ensure_schema()
    unit_id = "ro.test.integration.h08_d2"
    record = _legal_unit(unit_id, raw_text="integration fixture")

    first = repository.upsert_legal_units([record])
    second = repository.upsert_legal_units([record])

    assert first.attempted == 1
    assert second.attempted == 1
    assert first.inserted + first.updated + first.unchanged == 1
    assert second.unchanged == 1


class FakeTransaction:
    def __init__(self, connection):
        self.connection = connection
        self.snapshot = None

    async def __aenter__(self):
        self.snapshot = (
            copy.deepcopy(self.connection.legal_units),
            copy.deepcopy(self.connection.legal_edges),
            copy.deepcopy(self.connection.legal_embeddings),
            copy.deepcopy(self.connection.import_runs),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            (
                self.connection.legal_units,
                self.connection.legal_edges,
                self.connection.legal_embeddings,
                self.connection.import_runs,
            ) = self.snapshot
            self.connection.rolled_back = True
        else:
            self.connection.committed = True
        return False


class FakeConnection:
    def __init__(self, *, fail_on_edge: bool = False, fail_on_embedding: bool = False):
        self.fail_on_edge = fail_on_edge
        self.fail_on_embedding = fail_on_embedding
        self.legal_units = {}
        self.legal_edges = {}
        self.legal_embeddings = {}
        self.import_runs = {}
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def transaction(self):
        return FakeTransaction(self)

    async def close(self):
        self.closed = True

    async def execute(self, statement):
        return "OK"

    async def fetchrow(self, sql, *params):
        if "INSERT INTO legal_units" in sql:
            return self._upsert(self.legal_units, params)
        if "INSERT INTO legal_edges" in sql:
            if self.fail_on_edge:
                raise RuntimeError("edge upsert failed")
            return self._upsert(self.legal_edges, params)
        if "INSERT INTO legal_embeddings" in sql:
            if self.fail_on_embedding:
                raise RuntimeError("embedding upsert failed")
            return self._upsert_embedding(self.legal_embeddings, params)
        if "INSERT INTO import_runs" in sql and len(params) == 6:
            import_run_id, source_dir, mode, counts, warnings, errors = params
            self.import_runs[import_run_id] = {
                "source_dir": source_dir,
                "mode": mode,
                "status": "running",
                "counts": counts,
                "warnings": warnings,
                "errors": errors,
            }
            return {"id": import_run_id, "status": "running"}
        if "INSERT INTO import_runs" in sql and len(params) == 5:
            import_run_id, status, counts, warnings, errors = params
            self.import_runs.setdefault(import_run_id, {})
            self.import_runs[import_run_id].update(
                {
                    "status": status,
                    "counts": counts,
                    "warnings": warnings,
                    "errors": errors,
                }
            )
            return {"id": import_run_id, "status": status}
        raise AssertionError("unexpected SQL")

    @staticmethod
    def _upsert(table, params):
        record_id = params[0]
        existing = table.get(record_id)
        if existing is None:
            table[record_id] = params
            return {"inserted": True}
        if existing == params:
            return None
        table[record_id] = params
        return {"inserted": False}

    @staticmethod
    def _upsert_embedding(table, params):
        key = (params[0], params[3], params[5])
        existing = table.get(key)
        if existing is None:
            table[key] = params
            return {"inserted": True}
        if existing == params:
            return None
        table[key] = params
        return {"inserted": False}


def _write_bundle(tmp_path: Path, *, with_embeddings: bool = False) -> Path:
    source_dir = tmp_path / "bundle"
    source_dir.mkdir()
    _write_json(source_dir / "legal_units.json", [_legal_unit("ro.test"), _legal_unit("ro.test.art_1")])
    _write_json(source_dir / "legal_edges.json", [_legal_edge("edge.1", "ro.test", "ro.test.art_1")])
    _write_json(source_dir / "reference_candidates.json", [])
    _write_json(source_dir / "legal_chunks.json", [])
    _write_json(source_dir / "validation_report.json", {"import_blocking_passed": True})
    _write_json(source_dir / "corpus_manifest.json", {"sources": []})
    if with_embeddings:
        _write_jsonl(source_dir / "embeddings_input.jsonl", [_embedding_input_record()])
        _write_jsonl(source_dir / "embeddings_output.jsonl", [_embedding_record()])
        _write_json(
            source_dir / "validated_embeddings_manifest.json",
            {
                "input_path": str(source_dir / "embeddings_input.jsonl"),
                "output_path": str(source_dir / "embeddings_output.jsonl"),
                "model_name": "fake-model",
                "embedding_dim": 2,
                "input_read_count": 1,
                "output_read_count": 1,
                "embeddable_input_count": 1,
                "matched_output_count": 1,
                "missing_output_count": 0,
                "orphan_output_count": 0,
                "law_ids": ["ro.test"],
                "model_names": ["fake-model"],
                "embedding_dims": [2],
                "ready_for_pgvector_import": True,
                "validated_at": "2026-04-26T00:00:00Z",
                "validator_version": "test",
                "warnings": [],
            },
        )
    return source_dir


def _legal_unit(unit_id: str, *, raw_text: str = "Art. 1 text") -> dict:
    return {
        "id": unit_id,
        "canonical_id": unit_id.replace("ro.test.", "test:"),
        "source_id": "fixture_source",
        "law_id": "ro.test",
        "law_title": "Lege test",
        "status": "unknown",
        "hierarchy_path": ["Lege test"],
        "raw_text": raw_text,
        "normalized_text": raw_text,
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


def _embedding_input_record() -> dict:
    return {
        "record_id": "embedding.chunk.ro.test.art_1.0",
        "chunk_id": "chunk.ro.test.art_1.0",
        "legal_unit_id": "ro.test.art_1",
        "law_id": "ro.test",
        "text": "SECRET_RAW_TEXT",
        "embedding_text": "SECRET_EMBEDDING_TEXT",
        "text_hash": "hash-1",
        "metadata": {"retrieval_text_is_citable": False},
    }


def _embedding_record(*, embedding: list[float] | None = None) -> dict:
    return {
        "record_id": "embedding.chunk.ro.test.art_1.0",
        "chunk_id": "chunk.ro.test.art_1.0",
        "legal_unit_id": "ro.test.art_1",
        "law_id": "ro.test",
        "model_name": "fake-model",
        "embedding_dim": len(embedding or [0.1, 0.2]),
        "text_hash": "hash-1",
        "embedding": embedding or [0.1, 0.2],
        "metadata": {"retrieval_text_is_citable": False},
        "source_path": "fixture/embeddings_output.jsonl",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
