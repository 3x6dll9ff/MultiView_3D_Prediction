# Morphology RAG Base

This directory contains the initial curated RAG base for morphology-only explanation in `TriView3D`.

## Files

- `morphology_sources.jsonl`: chunk-ready records for retrieval

## Record format

Each line in `morphology_sources.jsonl` is a JSON object with these fields:

- `id`: unique chunk id
- `source_id`: stable source group id
- `title`: short source title
- `source_type`: source kind
- `url`: canonical source URL
- `priority`: `high` or `medium`
- `topics`: retrieval tags
- `chunk_type`: role of the chunk in prompting
- `content`: main grounded statement for retrieval
- `limitations`: safety note or scope constraint
- `follow_up`: suggested safe usage or next-step guidance

## Authoring rules

When extending this base:

1. Keep chunks short and specific.
2. Prefer morphology-first language.
3. Avoid mutation-level or treatment-level claims unless the source explicitly supports a cautious statement.
4. Add limitation text for every chunk.
5. Split metric definitions from biological interpretation when possible.

## Suggested chunk types

- `metric_definition`
- `morphology_summary`
- `biological_interpretation`
- `limitations`
- `feature_inventory`
- `plain_language_definition`
- `ai_explanation_template`

## Intended use

These chunks are designed for retrieval before calling an LLM so the model can explain:

- why a cell was flagged as anomalous
- which geometric or surface features matter
- what morphology may suggest biologically
- why morphology alone is insufficient for mutation or treatment claims
