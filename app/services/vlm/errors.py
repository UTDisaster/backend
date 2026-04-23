from __future__ import annotations


class VLMClassifyError(Exception):
    """Base error for VLM classification failures."""


class VLMRateLimitError(VLMClassifyError):
    def __init__(self, retry_after_seconds: float | None = None) -> None:
        super().__init__("VLM rate limited")
        self.retry_after_seconds = retry_after_seconds


class VLMParseError(VLMClassifyError):
    """Model returned output that could not be parsed into a VLMResult."""


class VLMFatalError(VLMClassifyError):
    """Non-retryable failure (bad request, invalid image, etc.)."""
