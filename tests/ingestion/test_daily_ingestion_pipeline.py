import json
from pathlib import Path

import pytest

import scripts.run_daily_ingestion_pipeline as daily_pipeline


def test_load_legal_sources_reads_valid_json(tmp_path):
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])

    sources = daily_pipeline.load_legal_sources(sources_path)

    assert sources == [
        {
            "source_id": "source_ro_codul_muncii",
            "law_id": "ro.codul_muncii",
            "law_title": "Codul muncii",
            "source_url": "https://legislatie.just.ro/Public/DetaliiDocument/128647",
            "status": "active",
            "enabled": True,
        }
    ]


def test_run_daily_ingestion_skips_disabled_sources(tmp_path, monkeypatch):
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)

    sources_path = _write_sources(
        tmp_path,
        [
            _source("ro.codul_muncii", enabled=True),
            _source("ro.codul_civil", enabled=False),
        ],
    )
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_total"] == 2
    assert summary["sources_enabled"] == 1
    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"] == 0
    assert [call["law_id"] for call in calls] == ["ro.codul_muncii"]


def test_dry_run_does_not_write_output_or_call_pipeline(tmp_path, monkeypatch):
    def fail_run_pipeline(**kwargs):
        raise AssertionError("dry-run must not call run_pipeline")

    output_root = tmp_path / "output"
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fail_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=output_root,
        dry_run=True,
    )

    assert summary["sources_total"] == 1
    assert summary["sources_enabled"] == 1
    assert summary["sources_succeeded"] == 0
    assert summary["sources_failed"] == 0
    assert summary["output_dirs"] == [str(output_root / "ro_codul_muncii")]
    assert not output_root.exists()


def test_output_dir_is_deterministic(tmp_path, monkeypatch):
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)

    output_root = tmp_path / "output"
    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=output_root,
    )

    expected = output_root / "ro_codul_muncii"
    assert calls[0]["out_dir"] == expected
    assert summary["output_dirs"] == [str(expected)]


def test_invalid_source_produces_clear_error(tmp_path):
    sources_path = _write_sources(
        tmp_path,
        [
            {
                "source_id": "broken",
                "law_id": "ro.broken",
                "law_title": "Broken",
                "source_url": "not-a-url",
                "enabled": True,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"source\[0\]\.source_url must be an http\(s\) URL"):
        daily_pipeline.load_legal_sources(sources_path)


def test_orchestrator_can_be_tested_with_monkeypatched_pipeline(tmp_path, monkeypatch):
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)

    sources_path = _write_sources(tmp_path, [_source("ro.codul_muncii")])
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
        write_debug=True,
    )

    assert len(calls) == 1
    assert calls[0]["url"] == "https://legislatie.just.ro/Public/DetaliiDocument/128647"
    assert calls[0]["source_id"] == "source_ro_codul_muncii"
    assert calls[0]["status"] == "active"
    assert calls[0]["write_debug"] is True
    assert summary["errors"] == []


def test_summary_counts_success_and_failure(tmp_path, monkeypatch):
    def fake_run_pipeline(**kwargs):
        if kwargs["law_id"] == "ro.fail":
            raise RuntimeError("fixture failure")

    sources_path = _write_sources(
        tmp_path,
        [
            _source("ro.ok"),
            _source("ro.fail"),
            _source("ro.disabled", enabled=False),
        ],
    )
    monkeypatch.setattr(daily_pipeline, "run_pipeline", fake_run_pipeline)

    summary = daily_pipeline.run_daily_ingestion(
        sources_path=sources_path,
        output_root=tmp_path / "output",
    )

    assert summary["sources_total"] == 3
    assert summary["sources_enabled"] == 2
    assert summary["sources_succeeded"] == 1
    assert summary["sources_failed"] == 1
    assert len(summary["output_dirs"]) == 2
    assert summary["errors"] == [
        {
            "source_id": "source_ro_fail",
            "law_id": "ro.fail",
            "error": "fixture failure",
        }
    ]


def _write_sources(tmp_path: Path, sources: list[dict]) -> Path:
    path = tmp_path / "legal_sources.json"
    path.write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    return path


def _source(law_id: str, *, enabled: bool = True) -> dict:
    return {
        "source_id": f"source_{law_id.replace('.', '_')}",
        "law_id": law_id,
        "law_title": "Codul muncii",
        "source_url": "https://legislatie.just.ro/Public/DetaliiDocument/128647",
        "status": "active",
        "enabled": enabled,
    }
