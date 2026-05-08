from __future__ import annotations

import logging
import os

from app.env_loader import load_app_env

logger = logging.getLogger(__name__)

ENV_VARS = (
    "APP_ENV",
    "DATABASE_URL",
    "GEMINI_API_KEY",
    "IMAGE_CONTENT_BASE_URL",
    "PARSED_DATA_DIR",
    "VLM_DATASET_DIR",
    "CORS_ALLOW_ORIGIN_REGEX",
    "CORS_ALLOW_ORIGINS",
)

def get_app_env() -> str:
    load_app_env()
    env = (os.getenv("APP_ENV", "dev") or "dev").strip().lower()
    if env not in ("dev", "prod"):
        raise RuntimeError("APP_ENV must be 'dev' or 'prod'")
    return env


def get_gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY", "") or "").strip()


def get_database_url() -> str:
    load_app_env()
    return (os.getenv("DATABASE_URL", "") or "").strip()


def get_image_content_base_url() -> str:
    load_app_env()
    return (os.getenv("IMAGE_CONTENT_BASE_URL", "") or "").strip()


def validate_env() -> None:
    load_app_env()
    get_app_env()
    for name in ENV_VARS:
        if not os.getenv(name):
            logger.warning("env var %s is not set", name)

    missing_required: list[str] = []
    if not get_database_url():
        missing_required.append("DATABASE_URL")
    if not get_image_content_base_url():
        missing_required.append("IMAGE_CONTENT_BASE_URL")

    if missing_required:
        raise RuntimeError(
            f"missing required env vars: {', '.join(missing_required)}"
        )

    if not get_gemini_api_key():
        logger.warning(
            "GEMINI_API_KEY is not set; Gemini-backed chat/VLM features will be unavailable."
        )
