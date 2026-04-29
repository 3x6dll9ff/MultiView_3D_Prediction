"""
Gemini LLM integration for morphology analysis reports.

Two-agent architecture:
  1. Writer  — generates structured report from data + RAG context.
  2. Verifier — validates the draft against RAG sources, corrects overclaiming.

Uses the ``google-genai`` SDK with ``gemini-2.5-flash``.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client = None

MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """\
You are a senior morphology analysis expert for a 3D cell reconstruction system.
Produce a structured, professional morphological analysis report based on the provided data.

RULES:
1. Use morphology-first language. NEVER claim specific mutations, diagnoses, or treatments.
2. Use cautious phrasing: "consistent with", "may indicate", "associated with".
3. Reference retrieved scientific sources naturally. Use their content to support claims.
4. Include specific numeric values for every metric you discuss.
5. Acknowledge limitations of morphology-only analysis explicitly.
6. Write in English, professional scientific register.

METRIC THRESHOLDS:
- Surface Roughness >= 0.11 → elevated
- Convexity <= 0.965 → reduced
- Sphericity <= 0.84 → reduced
- Eccentricity >= 0.68 → elevated
- Volume >= 18000 vx → enlarged

OUTPUT: respond with a single JSON object:
{
  "summary": "1-2 sentence verdict",
  "classification_interpretation": "2-3 sentences on classifier decision",
  "key_deviations": [
    {"metric":"name","value":0.0,"threshold":0.0,"status":"abnormal|borderline|normal","interpretation":"..."}
  ],
  "normal_metrics": ["metric: value — note"],
  "evidence": ["Evidence statement citing a source title"],
  "limitations": ["Limitation statement"],
  "recommendation": "1-2 sentence next-step suggestion"
}"""

VERIFIER_SYSTEM_PROMPT = """\
You are a scientific verifier for morphology analysis reports.
Review the draft report against the provided RAG sources and actual metric values.

CHECKLIST:
1. OVERCLAIMING — soften any diagnostic / mutation / treatment claims.
2. NUMERIC ACCURACY — verify cited values match actual data.
3. SOURCE ALIGNMENT — ensure evidence is supported by provided RAG chunks.
4. COMPLETENESS — expand limitations if needed.
5. TONE — ensure cautious, professional language.

OUTPUT: respond with a single JSON object identical in structure to the draft
plus a "corrections" array listing every change you made (empty if none):
{
  "summary": "...",
  "classification_interpretation": "...",
  "key_deviations": [...],
  "normal_metrics": [...],
  "evidence": [...],
  "limitations": [...],
  "recommendation": "...",
  "corrections": ["What was changed and why"]
}"""

# ---------------------------------------------------------------------------
# Client initialisation (google-genai SDK)
# ---------------------------------------------------------------------------


def _get_client():
    """Lazy-init google-genai Client. Returns None if unavailable."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — LLM features disabled")
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialised (%s)", MODEL_NAME)
        return _client
    except Exception as e:
        logger.error("Failed to initialise Gemini client: %s", e)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_rag_context(chunks: list[dict[str, Any]]) -> str:
    """Format RAG chunks into a context block for the LLM."""
    if not chunks:
        return "No relevant scientific sources were retrieved."
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Untitled")
        source_type = chunk.get("source_type", "Unknown")
        content = chunk.get("content", "")
        limitations = chunk.get("limitations", "")
        follow_up = chunk.get("follow_up", "")
        block = f"[Source {i}] {title} ({source_type})\nContent: {content}"
        if limitations:
            block += f"\nLimitations: {limitations}"
        if follow_up:
            block += f"\nUsage guidance: {follow_up}"
        parts.append(block)
    return "\n---\n".join(parts)


_REPORT_DEFAULTS: dict[str, Any] = {
    "summary": "",
    "classification_interpretation": "",
    "key_deviations": [],
    "normal_metrics": [],
    "evidence": [],
    "limitations": [],
    "recommendation": "",
}


def _ensure_fields(report: dict[str, Any]) -> dict[str, Any]:
    for key, default in _REPORT_DEFAULTS.items():
        report.setdefault(key, default)
    return report


def _call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict[str, Any] | None:
    """Call Gemini via google-genai SDK and return parsed JSON."""
    client = _get_client()
    if client is None:
        return None

    from google.genai import types

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=16384,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    raw = response.text
    if not raw or not raw.strip():
        logger.error("Gemini returned empty response")
        return None

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s — raw[:%d]: %s", e, min(200, len(raw)), raw[:200])
        raise


# ---------------------------------------------------------------------------
# Writer agent
# ---------------------------------------------------------------------------


def generate_report(
    classification: dict[str, Any],
    morphology: dict[str, float],
    metrics: dict[str, float],
    chunks: list[dict[str, Any]],
    cell_type: str = "",
) -> dict[str, Any] | None:
    """Writer agent: generate structured morphology report via Gemini."""
    cls = classification.get("class", "Unknown")
    conf = classification.get("confidence", 0.0)
    morpho_lines = "\n".join(f"  - {k}: {v:.4f}" for k, v in morphology.items())
    metrics_lines = "\n".join(
        f"  - {k}: {v}" for k, v in metrics.items() if v is not None
    )
    rag_context = _build_rag_context(chunks)
    cell_info = f"Cell type (from filename): {cell_type}\n" if cell_type else ""

    user_prompt = (
        f"CELL ANALYSIS DATA:\n{cell_info}"
        f"Classification: {cls} (confidence: {conf:.1%})\n\n"
        f"Morphometric Features:\n{morpho_lines}\n\n"
        f"Reconstruction Quality Metrics:\n{metrics_lines}\n\n"
        f"SCIENTIFIC KNOWLEDGE BASE (RAG):\n{rag_context}\n\n"
        "Generate the structured morphological analysis report."
    )

    try:
        report = _call_gemini(WRITER_SYSTEM_PROMPT, user_prompt, temperature=0.3)
        if report is None:
            return None
        return _ensure_fields(report)
    except Exception as e:
        logger.error("Writer agent failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Verifier agent
# ---------------------------------------------------------------------------


def verify_report(
    draft: dict[str, Any],
    chunks: list[dict[str, Any]],
    morphology: dict[str, float],
) -> dict[str, Any] | None:
    """Verifier agent: validate and correct the draft report via Gemini."""
    rag_context = _build_rag_context(chunks)
    morpho_lines = "\n".join(f"  - {k}: {v:.4f}" for k, v in morphology.items())

    user_prompt = (
        f"DRAFT REPORT TO VERIFY:\n{json.dumps(draft, indent=2)}\n\n"
        f"ACTUAL MORPHOMETRIC VALUES:\n{morpho_lines}\n\n"
        f"SCIENTIFIC KNOWLEDGE BASE:\n{rag_context}\n\n"
        "Verify the draft. Output the corrected report."
    )

    try:
        result = _call_gemini(VERIFIER_SYSTEM_PROMPT, user_prompt, temperature=0.1)
        if result is None:
            return None
        result.setdefault("corrections", [])
        return _ensure_fields(result)
    except Exception as e:
        logger.error("Verifier agent failed: %s", e)
        return None


def is_available() -> bool:
    """Check if LLM features are available."""
    return _get_client() is not None
