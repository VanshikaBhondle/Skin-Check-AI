"""Groq as a free AI provider (open vision models such as Llama 4), same JSON report as the other providers.

Get a free key at https://console.groq.com -> API Keys (starts with gsk_).
"""
from __future__ import annotations

import base64
import json
import os

MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
# Image-capable model name hints, best first; used if the configured model isn't available.
VISION_HINTS = ("llama-4-maverick", "llama-4-scout", "llama-4", "vision", "vl", "multimodal")
_resolved_model: str | None = None


def api_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def get_client():
    if not api_key():
        return None
    import groq
    return groq.Groq(api_key=api_key(), max_retries=2, timeout=90)


def _pick_vision_model(client, exclude: str) -> str | None:
    ids = [m.id for m in client.models.list().data if m.id != exclude]
    for hint in VISION_HINTS:
        for i in sorted(ids, reverse=True):
            if hint in i.lower():
                return i
    return None


def _is_model_problem(e) -> bool:
    import groq
    if isinstance(e, groq.NotFoundError):
        return True
    if isinstance(e, groq.BadRequestError):
        m = str(e).lower()
        return any(w in m for w in ("decommission", "does not exist", "not found", "image", "vision", "content must be a string"))
    return False


def _friendly(e, AnalysisError):
    import groq
    msg = str(getattr(e, "message", "") or e)[:200]
    if isinstance(e, groq.AuthenticationError):
        return AnalysisError("Groq rejected the API key. Check GROQ_API_KEY in Secrets.")
    if isinstance(e, groq.PermissionDeniedError):
        return AnalysisError(f"Groq refused this request (403). Groq says: {msg}")
    if isinstance(e, groq.RateLimitError):
        return AnalysisError("The free Groq limit is reached for now. Please try again in a minute.")
    if isinstance(e, groq.APIStatusError):
        if e.status_code >= 500:
            return AnalysisError("Groq had a temporary problem. Please try again.")
        return AnalysisError(f"The Groq request failed ({e.status_code}). Groq says: {msg}")
    if isinstance(e, (groq.APIConnectionError, groq.APITimeoutError)):
        return AnalysisError("Couldn't reach Groq. Please try again.")
    return AnalysisError("The Groq request failed. Please try again.")


def chat(client, messages: list, AnalysisError, json_mode: bool = False, max_tokens: int = 4096) -> tuple[str, str]:
    """One chat completion with automatic vision-model fallback and friendly errors. Returns (text, model)."""
    global _resolved_model
    import groq
    model = _resolved_model or MODEL
    kwargs = {"messages": messages, "max_completion_tokens": max_tokens, "temperature": 0.2}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        try:
            resp = client.chat.completions.create(model=model, **kwargs)
        except groq.APIStatusError as e:
            if not _is_model_problem(e):
                raise
            fallback = _pick_vision_model(client, exclude=model)
            if not fallback:
                raise
            _resolved_model = model = fallback
            resp = client.chat.completions.create(model=model, **kwargs)
    except AnalysisError:
        raise
    except Exception as e:
        raise _friendly(e, AnalysisError) from e
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise AnalysisError("The AI response was cut off. Please try again.")
    text = choice.message.content or ""
    if not text.strip():
        raise AnalysisError("Groq returned an empty answer. Try a clear close-up photo of the skin area.")
    return text, model


def _image_part(jpeg: bytes) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("ascii")}}


def analyze(jpeg: bytes, prompt: str, system: str, schema: dict, AnalysisError, client=None) -> tuple[dict, str]:
    client = client or get_client()
    instructions = (system + "\n\nReply with ONE JSON object only (no markdown, no extra text) that follows this "
                    "JSON Schema exactly, with every required field present:\n" + json.dumps(schema))
    text, model = chat(client, [
        {"role": "system", "content": instructions},
        {"role": "user", "content": [{"type": "text", "text": prompt}, _image_part(jpeg)]},
    ], AnalysisError, json_mode=True)
    text = text.strip()
    if text.startswith("```"):  # tolerate a fenced reply
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text), model
    except json.JSONDecodeError as e:
        raise AnalysisError("The AI returned an unreadable answer. Please try again.") from e


def start_chat(jpeg: bytes, context: str, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": context}, _image_part(jpeg)]},
        {"role": "assistant", "content": "Thanks - I have the photo and the analysis. What would you like to know?"},
    ]


def is_history(history) -> bool:
    return bool(history) and isinstance(history[0], dict) and history[0].get("role") == "system"


def ask(history: list[dict], question: str, AnalysisError, client=None) -> str:
    client = client or get_client()
    if client is None:
        raise AnalysisError("AI chat isn't configured: add GROQ_API_KEY to the app's Secrets.")
    history.append({"role": "user", "content": question})
    try:
        text, _ = chat(client, history, AnalysisError, max_tokens=1024)
    except AnalysisError:
        history.pop()  # keep the history valid so the person can retry
        raise
    history.append({"role": "assistant", "content": text})
    return text
