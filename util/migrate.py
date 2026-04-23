from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def _applied(conn) -> set[str]:
    rows = conn.execute(text("SELECT name FROM schema_migrations")).all()
    return {row[0] for row in rows}


def _discover() -> list[Path]:
    if not MIGRATIONS_DIR.is_dir():
        return []
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.is_file())


def main() -> int:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 1

    engine = create_engine(db_url, future=True)
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

    files = _discover()
    with engine.connect() as conn:
        already = _applied(conn)

    for path in files:
        if path.name in already:
            continue
        sql = path.read_text(encoding="utf-8")
        with engine.begin() as conn:
            conn.execute(text(sql))
            conn.execute(
                text("INSERT INTO schema_migrations (name) VALUES (:name)"),
                {"name": path.name},
            )
        print(f"applied {path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
