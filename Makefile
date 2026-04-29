# Backend dev commands. Run `make dev` after copying .env.example to .env.

.PHONY: dev db seed test eval lint migrate

.env:
	cp .env.example .env
	@echo ">>> Created .env from .env.example. Fill in GEMINI_API_KEY and rerun 'make dev'."
	@false

dev: .env
	docker compose up --build

db:
	docker compose up -d db

seed:
	@echo "No seed script is provided. Use preprocess-data load instead:"
	@echo "  DATABASE_URL='postgresql+psycopg://utd:utdpass@127.0.0.1:5433/utd_data' \\"
	@echo "  python util/preprocess-data.py --start-at load --stop-after load --input <path-to-parsed_data.json>"
	@false

test:
	pytest

eval:
	python util/vlm_eval.py --disaster-id hurricane-florence

lint:
	ruff check . && black --check .

migrate:
	python util/migrate.py
