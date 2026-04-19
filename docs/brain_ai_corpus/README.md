# Brain AI Corpus

This folder contains a filtered project-memory corpus built from the external brain archive.

Selection rules:
- include markdown artifacts only
- exclude .resolved variants, metadata, browser scratch files, cache folders, and content stubs
- keep the latest version for each artifact filename
- sort output chronologically by last write time

Artifacts kept: 69
Generated on: 2026-04-19 17:18:54

Files:
- timeline.json: machine-friendly manifest
- timeline.csv: spreadsheet-friendly manifest
- artifacts/: cleaned markdown corpus

Recommended ingestion order:
1. docs/brain_ai_ingestion_summary.md
2. timeline.json or timeline.csv
3. the files in artifacts/ from newest to oldest or by topic

- project_memory.md: single-file direct LLM context pack
- timeline.jsonl: one JSON object per artifact with full content

