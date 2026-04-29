# Backend

## Quick Start

```bash
cp .env.example .env
# edit .env and fill in GEMINI_API_KEY (and update other values as needed)
make dev
```

Once the stack is up, `GET http://localhost:8000/health` should return 200.

## Environment

All environment variables the backend reads are listed in [`.env.example`](./.env.example). That file is the canonical reference — one variable per line, grouped by concern (runtime mode, database, Gemini, image content, preprocessing, CORS).

## Common tasks

- `make dev` — build and run the full stack (api + db) via docker compose
- `make db` — start only the Postgres/PostGIS container
- `make test` — run pytest
- `make eval` — run the VLM evaluation against `hurricane-florence`
- `make lint` — run `ruff check .` and `black --check .`

---

For the non-Docker flow (manual uv + raw `docker run`), see [docs/manual-setup.md](./docs/manual-setup.md).
