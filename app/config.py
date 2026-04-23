from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_VARS = (
    "DATABASE_URL",
    "GEMINI_API_KEY",
    "PARSED_DATA_DIR",
    "SUPABASE_URL",
    "SUPABASE_IMAGES_BUCKET",
    "SUPABASE_STRIP_PREFIX",
    "SUPABASE_USE_BASENAME",
    "SUPABASE_SERVICE_ROLE_KEY",
    "CORS_ALLOW_ORIGIN_REGEX",
    "CORS_ALLOW_ORIGINS",
)

REQUIRED = ("DATABASE_URL", "GEMINI_API_KEY")


def validate_env() -> None:
    for name in ENV_VARS:
        if not os.getenv(name):
            logger.warning("env var %s is not set", name)

    missing_required = [name for name in REQUIRED if not os.getenv(name)]
    if missing_required:
        raise RuntimeError(
            f"missing required env vars: {', '.join(missing_required)}"
        )
