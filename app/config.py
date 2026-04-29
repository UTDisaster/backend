from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_VARS = (
    "APP_ENV",
    "DATABASE_URL",
    "DEV_DATABASE_URL",
    "GEMINI_API_KEY",
    "DEV_GEMINI_API_KEY",
    "IMAGE_CONTENT_BASE_URL",
    "DEV_IMAGE_CONTENT_BASE_URL",
    "PARSED_DATA_DIR",
    "VLM_DATASET_DIR",
    "CORS_ALLOW_ORIGIN_REGEX",
    "CORS_ALLOW_ORIGINS",
)

def get_app_env() -> str:
    env = (os.getenv("APP_ENV", "dev") or "dev").strip().lower()
    if env not in ("dev", "prod"):
        raise RuntimeError("APP_ENV must be 'dev' or 'prod'")
    return env


def get_gemini_api_key() -> str:
    app_env = get_app_env()
    if app_env == "dev":
        return (os.getenv("DEV_GEMINI_API_KEY", "") or "").strip()
    return (os.getenv("GEMINI_API_KEY", "") or "").strip()


def get_database_url() -> str:
    app_env = get_app_env()
    if app_env == "dev":
        return (os.getenv("DEV_DATABASE_URL", "") or "").strip()
    return (os.getenv("DATABASE_URL", "") or "").strip()


def get_image_content_base_url() -> str:
    app_env = get_app_env()
    if app_env == "dev":
        return (os.getenv("DEV_IMAGE_CONTENT_BASE_URL", "") or "").strip()
    return (os.getenv("IMAGE_CONTENT_BASE_URL", "") or "").strip()


def validate_env() -> None:
    app_env = get_app_env()
    for name in ENV_VARS:
        if not os.getenv(name):
            logger.warning("env var %s is not set", name)

    missing_required: list[str] = []
    if app_env == "dev":
        if not get_database_url():
            missing_required.append("DEV_DATABASE_URL")
        if not get_image_content_base_url():
            missing_required.append("DEV_IMAGE_CONTENT_BASE_URL")
        if not get_gemini_api_key():
            missing_required.append("DEV_GEMINI_API_KEY")
    else:
        if not get_database_url():
            missing_required.append("DATABASE_URL")
        if not get_image_content_base_url():
            missing_required.append("IMAGE_CONTENT_BASE_URL")
        if not get_gemini_api_key():
            missing_required.append("GEMINI_API_KEY")

    if missing_required:
        raise RuntimeError(
            f"missing required env vars: {', '.join(missing_required)}"
        )
