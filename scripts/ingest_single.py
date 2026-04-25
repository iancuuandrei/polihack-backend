"""
scripts/ingest_single.py
-----------------------
Single-URL ingestion orchestrator for the AntiGravity pipeline.

Usage:
    python scripts/ingest_single.py \
        --url "https://legislatie.just.ro/Public/DetaliiDocument/253966" \
        --law-id "ro.ordin_745_2020" \
        --out-dir ingestion/output/single_ingest_v1
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ingestion.parser.html_scraper import scrape_html_source
from ingestion.parser.html_parser import parse_printable_text, extract_metadata
from ingestion.parser.atomic_parser import AtomicParser
from ingestion.reference_resolver import resolve_references
from ingestion.validators import build_validation_report, save_validation_report, validate_corpus
from ingestion.manifest import build_manifest, save_manifest


def run_pipeline(url: Optional[str], file_path: Optional[str], law_id: str, out_dir: Path):
    print(f"\n=== AntiGravity Single Ingestion Pipeline ===")
    if url:
        print(f"URL     : {url}")
    if file_path:
        print(f"File    : {file_path}")
    print(f"Law ID  : {law_id}")
    print(f"Output  : {out_dir}\n")

    # 1. Get HTML Source
    html = None
    if url:
        print(f"[*] Scraping HTML source from URL...")
        html = scrape_html_source(url)
    elif file_path:
        print(f"[*] Reading HTML source from file...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            print(f"[FATAL] Failed to read file: {e}")
            sys.exit(1)

    if not html:
        print("[FATAL] No HTML content provided or found.")
        sys.exit(1)

    # 2. Extract Metadata & Clean Text
    print(f"[*] Parsing HTML to clean printable text...")
    metadata = extract_metadata(html)
    clean_text = parse_printable_text(html)
    if not clean_text:
        print("[FATAL] Failed to extract printable text from HTML.")
        sys.exit(1)
    
    law_title = metadata.get("title", law_id)
    print(f"    Detected Title: {law_title}")

    # 3. Atomic Parsing (Structure + Edges + Refs)
    print(f"[*] Running AtomicParser (generating units and hierarchy edges)...")
    parser = AtomicParser(corpus_id=law_id)
    units, contains_edges = parser.parse(clean_text)
    
    # 4. Extract Reference Candidates
    from ingestion.reference_extractor import extract_references
    all_candidates = []
    for unit in units:
        candidates = extract_references({"id": unit["id"], "raw_text": unit["text"]})
        all_candidates.extend(candidates)

    # 5. Resolve References (Intra-act)
    print(f"[*] Resolving internal references...")
    resolved_candidates, ref_edges = resolve_references(all_candidates, units)

    # 6. Validate
    print(f"[*] Validating corpus integrity...")
    try:
        validate_corpus(units)
    except ValueError as exc:
        print(f"[WARN] Validation failed: {exc}")
    
    report = build_validation_report(units, contains_edges, all_candidates)

    # 7. Save Bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_edges = contains_edges + ref_edges
    
    files = {
        "legal_units.json": units,
        "legal_edges.json": all_edges,
        "reference_candidates.json": resolved_candidates,
        "validation_report.json": report,
    }

    for filename, data in files.items():
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # 8. Build Manifest
    sources_mock = [{
        "law_id": law_id,
        "law_title": law_title,
        "url": url or file_path
    }]
    manifest = build_manifest(out_dir.name, sources_mock, out_dir)
    save_manifest(manifest, out_dir / "corpus_manifest.json")

    print(f"\n=== Success! Output written to {out_dir} ===")


def main():
    parser = argparse.ArgumentParser(description="Ingest a single law from a URL or HTML file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="The URL to scrape")
    group.add_argument("--file", help="Path to a local HTML file")
    
    parser.add_argument("--law-id", required=True, help="Deterministic ID (e.g. ro.lege_123_2023)")
    parser.add_argument("--out-dir", required=True, help="Where to save the JSON bundle")
    
    args = parser.parse_args()
    run_pipeline(args.url, args.file, args.law_id, Path(args.out_dir))


if __name__ == "__main__":
    main()
