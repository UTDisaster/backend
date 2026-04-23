This is the manual setup path for contributors who cannot use Docker Compose. Prefer the Quick Start in the main README.

# Backend

## Setup

Recommended to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage virtual environment and pip packages.

### First Installation

Create virtual environment:

```bash
uv venv
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

## PostgreSQL + PostGIS

Note: The following names and passwords are only for development purposes, and will be different in production.

Create a PostgreSQL database with PostGIS enabled, then set:

```bash
docker run --name utd-postgis \
    -e POSTGRES_USER=utd \
    -e POSTGRES_PASSWORD=utdpass \
    -e POSTGRES_DB=utd_data \
    -p 5433:5432 \
    -d postgis/postgis:16-3.4
```

If this not the first time running, use now:

```bash
docker start utd-postgis
```

Make sure your environment has the variable DATABASE_URL which corresponds to the new DB.

```bash
export DATABASE_URL='postgresql+psycopg://utd:utdpass@localhost:5433/utd_data'
```

## Preprocessing Pipeline

Run both parsing and DB loading in one command:

```bash
python3 util/preprocess-data.py ../test_images_labels_targets/test --output data-example
```

## Running the API

If parsed images are not under `data-example`, point the API to the parser output root:

```bash
export PARSED_DATA_DIR='data-example'
```

```bash
uvicorn app.main:app --reload
```
