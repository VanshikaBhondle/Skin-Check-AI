"""Google Gemini as a free alternative AI provider (same JSON report as the Claude path).

Used when GEMINI_API_KEY (or GOOGLE_API_KEY) is set and ANTHROPIC_API_KEY is not.
Get a free key at https://aistudio.google.com -> Get API key.
"""
from __future__ import annotations

import copy
import json
import os

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")  # alias that tracks Google's current Flash model
_resolved_model: str | None = None  # remembered after an automatic fallback


def api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def get_client():
    if not api_key():
        return None
    from google import genai
    return genai.Client(api_key=api_key())


def _strip(schema):
    """Gemini's schema support is narrower than Claude's - drop keywords it may reject."""
    if isinstance(schema, dict):
        return {k: _strip(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip(v) for v in schema]
    return schema


def _friendly(e: Exception, AnalysisError):
    from google.genai import errors
    if isinstance(e, errors.ClientError):
        msg = str(getattr(e, "message", "") or e)
        reason = f" Google says: {msg[:200]}" if msg else ""
        if e.code == 429:
            return AnalysisError("The free Gemini limit is reached for now. Please try again in a few minutes.")
        if e.code in (401, 403) or "API key" in msg or "API_KEY" in msg:
            return AnalysisError(f"Google rejected the API key ({e.code}). Check GEMINI_API_KEY in Secrets.{reason}")
        return AnalysisError(f"The Gemini request failed ({e.code}).{reason}")
    if isinstance(e, errors.ServerError):
        return AnalysisError("Gemini had a temporary problem. Please try again.")
    return AnalysisError("Couldn't reach Gemini. Check the internet connection.")


def _pick_flash_model(client) -> str | None:
    """Fallback if the configured model name doesn't exist: pick an available Flash model."""
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions and "flash" in (m.name or "") and "image" not in (m.name or ""):
            names.append(m.name.removeprefix("models/"))
    stable = [n for n in names if "preview" not in n and "exp" not in n and "lite" not in n]
    return sorted(stable or names, reverse=True)[0] if names else None


def generate(client, contents, config: dict, AnalysisError):
    """One generate_content call with model fallback, safety checks and friendly errors. Returns (text, model)."""
    global _resolved_model
    from google.genai import errors, types
    model = _resolved_model or MODEL
    try:
        try:
            resp = client.models.generate_content(model=model, contents=contents,
                                                  config=types.GenerateContentConfig(**config))
        except errors.ClientError as e:
            if e.code != 404:
                raise
            fallback = _pick_flash_model(client)
            if not fallback:
                raise
            _resolved_model = model = fallback
            resp = client.models.generate_content(model=model, contents=contents,
                                                  config=types.GenerateContentConfig(**config))
    except AnalysisError:
        raise
    except Exception as e:
        raise _friendly(e, AnalysisError) from e

    fb = getattr(resp, "prompt_feedback", None)
    if fb is not None and getattr(fb, "block_reason", None):
        raise AnalysisError("Gemini declined to analyse this image. Try a clear close-up photo of the skin area.")
    cand = (resp.candidates or [None])[0]
    reason = str(getattr(cand, "finish_reason", "") or "")
    if "MAX_TOKENS" in reason:
        raise AnalysisError("The AI response was cut off. Please try again.")
    text = resp.text or ""
    if not text.strip():
        raise AnalysisError("Gemini declined or returned an empty answer. Try a clear close-up photo of the skin area.")
    return text, model


def analyze(jpeg: bytes, prompt: str, system: str, schema: dict, AnalysisError, client=None) -> tuple[dict, str]:
    from google.genai import types
    client = client or get_client()
    text, model = generate(
        client,
        [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), types.Part.from_text(text=prompt)],
        {"system_instruction": system, "response_mime_type": "application/json",
         "response_json_schema": _strip(copy.deepcopy(schema)), "max_output_tokens": 8192},
        AnalysisError)
    try:
        return json.loads(text), model
    except json.JSONDecodeError as e:
        raise AnalysisError("The AI returned an unreadable answer. Please try again.") from e


def start_chat(jpeg: bytes, context: str) -> list:
    from google.genai import types
    return [
        types.Content(role="user", parts=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                                          types.Part.from_text(text=context)]),
        types.Content(role="model", parts=[types.Part.from_text(
            text="Thanks - I have the photo and the analysis. What would you like to know?")]),
    ]


def is_history(history) -> bool:
    return bool(history) and not isinstance(history[0], dict)


def ask(history: list, question: str, system: str, AnalysisError, client=None) -> str:
    from google.genai import types
    client = client or get_client()
    if client is None:
        raise AnalysisError("AI chat isn't configured: add GEMINI_API_KEY to the app's Secrets.")
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    try:
        text, _ = generate(client, history, {"system_instruction": system, "max_output_tokens": 2048}, AnalysisError)
    except AnalysisError:
        history.pop()  # keep the history valid so the person can retry
        raise
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
    return text
