from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_LOADED = False


def load_app_env() -> None:
    """Load env vars using .env as default and .env.prod as prod overlay.

    Behavior:
    - Load backend/.env first (default values; no override).
    - If APP_ENV=prod, load backend/.env.prod with override enabled.
    """
    global _LOADED
    if _LOADED:
        return

    root = Path(__file__).resolve().parents[1]
    selector_file = root / ".env"
    if selector_file.is_file():
        load_dotenv(selector_file, override=False)

    app_env = (os.getenv("APP_ENV", "dev") or "dev").strip().lower()
    if app_env not in ("dev", "prod"):
        raise RuntimeError("APP_ENV must be 'dev' or 'prod'")

    if app_env == "prod":
        prod_file = root / ".env.prod"
        if prod_file.is_file():
            load_dotenv(prod_file, override=True)

    _LOADED = True
