"""LLM client: OpenAI-compatible chat.completions with a JSON-object reply validated by pydantic.

chat.completions (not the Responses API) so any OpenAI-compatible endpoint works (OpenAI, NVIDIA NIM, local).
Primary LLM_* and optional LLM_FALLBACK_* providers; no keys or DEMO_MODE=1 -> callers use their rule-based path.
Only aggregates go to the LLM: metrics, column names, counts — never raw rows of the organizers' data.
"""

import json
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, PermissionDeniedError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# .env.local wins on conflicts since it loads first and override=False keeps already-set values.
load_dotenv(_PROJECT_ROOT / ".env.local", override=False)
load_dotenv(_PROJECT_ROOT / ".env", override=False)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_LAST: ContextVar[dict | None] = ContextVar("last_llm_provider", default=None)


def _demo_forced() -> bool:
    return os.getenv("DEMO_MODE", "auto").strip().lower() in {"1", "true"}


def _providers() -> list[tuple[str, str, str]]:
    """(api_key, base_url, model) in priority order: primary LLM_*, then optional LLM_FALLBACK_*."""
    out = []
    for prefix in ("LLM", "LLM_FALLBACK"):
        key, model = os.getenv(f"{prefix}_API_KEY"), os.getenv(f"{prefix}_MODEL")
        if key and model:
            out.append((key, os.getenv(f"{prefix}_BASE_URL", DEFAULT_BASE_URL), model))
    return out


def llm_mode() -> Literal["llm", "demo"]:
    if _demo_forced():
        return "demo"
    return "llm" if _providers() else "demo"


def last_provider() -> dict:
    """Which provider answered the last successful call (for the agent log)."""
    return dict(_LAST.get() or {})


def _extract_json(content: str) -> dict:
    """Some OpenAI-compatible endpoints wrap JSON in prose or code fences."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def complete_json[ModelT: BaseModel](
    model_cls: type[ModelT], system: str, user: str, budget_s: float = 30
) -> ModelT | None:
    """Call the LLM(s) and validate the JSON reply against model_cls, or return None."""
    _LAST.set({})
    providers = [] if _demo_forced() else _providers()
    if not providers:
        return None
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    deadline = time.monotonic() + budget_s
    for api_key, base_url, model in providers:
        for fmt in ({"type": "json_object"}, None):  # not every endpoint supports json mode
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                kwargs = {"model": model, "messages": messages, "max_tokens": 1200}
                if fmt:
                    kwargs["response_format"] = fmt
                with OpenAI(api_key=api_key, base_url=base_url, timeout=min(20, remaining),
                            max_retries=0) as client:
                    response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty completion content")
                result = model_cls.model_validate(_extract_json(content))
                usage = getattr(response, "usage", None)
                _LAST.set(dict(provider=base_url, model=model,
                               tokens=getattr(usage, "total_tokens", 0) or 0))
                return result
            except (AuthenticationError, PermissionDeniedError) as exc:
                # A different JSON format cannot repair rejected credentials or permissions.
                logger.warning("complete_json failed (%s): %s", model, type(exc).__name__)
                break
            except Exception as exc:  # invalid format -> next attempt / provider / rules
                logger.warning("complete_json failed (%s): %s", model, type(exc).__name__)
    return None
