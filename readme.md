# Backend

## Setup

Recommended to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage virtual environment and pip packages.

### TLDR

First setup:

```bash
uv venv

source .venv/bin/activate

uv pip install -r requirements.txt

docker run --name utd-postgis \
    -e POSTGRES_USER=utd \
    -e POSTGRES_PASSWORD=utdpass \
    -e POSTGRES_DB=utd_data \
    -p 5432:5432 \
    -d postgis/postgis:16-3.4
    
export DATABASE_URL='postgresql+psycopg://utd:utdpass@localhost:5432/utd_data'

python3 util/preprocess-data.py ../test_images_labels_targets/test --output data-example

uvicorn app.main:app --reload
```

Subsequent runs:

```bash
docker start utd-postgis
    
export DATABASE_URL='postgresql+psycopg://utd:utdpass@localhost:5432/utd_data'

python3 util/preprocess-data.py ../test_images_labels_targets/test --output data-example

uvicorn app.main:app --reload
```

The 3rd command is only necessary if you made changes to the pipeline or database.

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
    -p 5432:5432 \
    -d postgis/postgis:16-3.4
```

If this not the first time running, use now:

```bash
docker start utd-postgis
```

Make sure your environment has the variable DATABASE_URL which corresponds to the new DB.

```bash
export DATABASE_URL='postgresql+psycopg://utd:utdpass@localhost:5432/utd_data'
```

## Preprocessing Pipeline

Run parsing, VLM prediction, and DB loading in one command:

```bash
python3 util/preprocess-data.py ../test_images_labels_targets/test --output data-example
```

### Fine-grained usage:

#### Data Parsing

```bash
# just generate an initial parsed_data.json file without vlm stuff
python3 util/preprocess-data.py ../test_images_labels_targets/test --output data-example --stop-after parse
```

#### Image Manipulation

Nothing here yet.

#### VLM Classification

```bash
# run only the vlm, on the existing parsed data
python3 util/preprocess-data.py --output data-example --start-at vlm --stop-after vlm

# same as above, but prints a table instead of writing to parsed_data.json
python3 util/preprocess-data.py --output data-example --start-at vlm --stop-after vlm --vlm-print table --vlm-no-write

# same as above command, but limits how many classifications to make (randomized selection)
python3 util/preprocess-data.py --output data-example --start-at vlm --stop-after vlm --vlm-print table --vlm-no-write --vlm-max-locations 25 --vlm-max-locations-per-image 5 --vlm-randomize
```

#### Loading Data to Database

```bash
# Load the data in parsed_data.json and images into the database
python3 util/preprocess-data.py --output data-example --start-at load
```

## Running the API

If parsed images are not under `data-example`, point the API to the parser output root:

```bash
export PARSED_DATA_DIR='data-example'
```

```bash
uvicorn app.main:app --reload
```
