# Backend

## Quick Start

```bash
cp .env.example .env
# edit .env values (DB URL, image base URL, keys)
make dev
```

Once the stack is up, `GET http://localhost:8000/health` should return 200.

## Environment

Env selection is controlled only by `APP_ENV`:

- `.env` is the default file (typically `APP_ENV=dev`).
- when `APP_ENV=prod`, backend overlays `.env.prod` values.

Use `.env.example` and `.env.prod.example` as templates. Both files use identical variable names.

## Common tasks

- `make dev` — build and run the full stack (api + db) via docker compose
- `make db` — start only the Postgres/PostGIS container
- `make test` — run pytest
- `make eval` — run the VLM evaluation against `hurricane-florence`
- `make lint` — run `ruff check .` and `black --check .`

---

For the non-Docker flow (manual uv + raw `docker run`), see [docs/manual-setup.md](./docs/manual-setup.md).
