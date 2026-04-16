from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from app.services.vlm.errors import (
    VLMFatalError,
    VLMParseError,
    VLMRateLimitError,
)
from app.services.vlm.parse import parse_response
from app.services.vlm.prompt import PROMPTS, Prompt, PromptVersion
from app.services.vlm.rate_limit import BackoffPolicy, TokenBucket

load_dotenv()


@dataclass(frozen=True)
class VLMResult:
    score: int
    label: str
    confidence: float
    description: str
    prompt_version: str
    model: str
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None


class VLMClassifier(Protocol):
    def classify_pair(
        self,
        pre: bytes,
        post: bytes,
        *,
        pair_id: str | None = None,
        location_uid: str | None = None,
    ) -> VLMResult: ...


@lru_cache(maxsize=1)
def _default_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise VLMFatalError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _extract_retry_after_seconds(message: str) -> float | None:
    retry_in = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if retry_in:
        return max(1.0, float(retry_in.group(1)))
    retry_delay = re.search(r"'retryDelay': '([0-9]+)s'", message, re.IGNORECASE)
    if retry_delay:
        return max(1.0, float(retry_delay.group(1)))
    return None


def _classify_error(exc: Exception) -> Exception:
    if isinstance(exc, (ClientError, ServerError, APIError)):
        message = str(exc)
        status_code = getattr(exc, "status_code", 0) or getattr(exc, "code", 0)
        if status_code == 429 or "RESOURCE_EXHAUSTED" in message or "rate" in message.lower():
            return VLMRateLimitError(_extract_retry_after_seconds(message))
        if status_code in (500, 502, 503, 504) or "UNAVAILABLE" in message:
            return VLMRateLimitError(None)
        return VLMFatalError(f"Gemini APIError {status_code}: {message[:200]}")
    return VLMFatalError(f"{type(exc).__name__}: {exc}")


class GeminiVLMClassifier:
    """Gemini Vision damage classifier.

    Respects:
    - Token-bucket rate limiting (single bucket across calls).
    - Exponential backoff on 429/5xx with optional retry-after hint.
    - One retry on parse failure, then fail.
    - Never retries fatal errors (400, auth, etc).
    """

    def __init__(
        self,
        client: genai.Client | None = None,
        *,
        model: str = "gemini-2.5-flash",
        prompt_version: PromptVersion = "v2",
        rate: TokenBucket | None = None,
        backoff: BackoffPolicy | None = None,
        temperature: float = 0.2,
    ) -> None:
        self._client = client if client is not None else _default_client()
        self._model = model
        self._prompt = PROMPTS[prompt_version]
        self._rate = rate if rate is not None else TokenBucket(rate_per_second=5.0)
        self._backoff = backoff if backoff is not None else BackoffPolicy()
        self._temperature = temperature

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    def classify_pair(
        self,
        pre: bytes,
        post: bytes,
        *,
        pair_id: str | None = None,
        location_uid: str | None = None,
    ) -> VLMResult:
        if not pre or not post:
            raise VLMFatalError("pre and post image bytes are required")

        prompt = self._prompt
        parts = self._build_parts(prompt, pre, post)

        last_parse_error: Exception | None = None
        for attempt in range(self._backoff.max_attempts):
            self._rate.acquire()
            started = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        temperature=self._temperature,
                        response_mime_type="application/json",
                        system_instruction=prompt.system_instruction,
                    ),
                )
            except VLMRateLimitError:
                raise
            except Exception as exc:
                classified = _classify_error(exc)
                if isinstance(classified, VLMRateLimitError):
                    time.sleep(
                        self._backoff.delay_for(attempt, classified.retry_after_seconds)
                    )
                    continue
                raise classified from exc

            latency_ms = int((time.monotonic() - started) * 1000)
            raw = getattr(response, "text", None) or self._collect_text(response)
            try:
                parsed = parse_response(raw)
            except VLMParseError as exc:
                last_parse_error = exc
                if attempt >= self._backoff.max_attempts - 1:
                    break
                time.sleep(self._backoff.delay_for(attempt))
                continue

            return VLMResult(
                score=parsed["score"],
                label=parsed["label"],
                confidence=parsed["confidence"],
                description=parsed["description"],
                prompt_version=prompt.version,
                model=self._model,
                latency_ms=latency_ms,
                tokens_in=_usage_field(response, "prompt_token_count"),
                tokens_out=_usage_field(response, "candidates_token_count"),
            )

        raise last_parse_error or VLMParseError("Exhausted attempts without a parseable response")

    @staticmethod
    def _build_parts(prompt: Prompt, pre: bytes, post: bytes) -> list[types.Part]:
        return [
            types.Part(text="PRE (before) satellite crop:"),
            types.Part.from_bytes(data=pre, mime_type="image/png"),
            types.Part(text="POST (after) satellite crop:"),
            types.Part.from_bytes(data=post, mime_type="image/png"),
            types.Part(text=prompt.user_instruction),
        ]

    @staticmethod
    def _collect_text(response: object) -> str:
        candidates = getattr(response, "candidates", None) or []
        texts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        return "".join(texts)


def _usage_field(response: object, name: str) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None
    value = getattr(usage, name, None)
    return int(value) if isinstance(value, int) else None
