from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    """Load environment variables from the common repo locations.

    The backend is often launched from the `backend/` directory, but this
    workspace also keeps a shared `.env` under `meta/`. We load both so local
    runs and containerized runs can work without manual exports.
    """

    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "backend" / ".env",
        repo_root / "meta" / ".env",
        repo_root / ".env",
    )

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
