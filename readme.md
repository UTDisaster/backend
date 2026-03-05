# Backend

## Info

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

## After Pulling Changes

Install possible new dependencies added by someone else:

```bash
uv pip install -r requirements.txt
```

## Running the Project for Development

### Running the Project

```bash
uvicorn app.main:app --reload
```

### Status Check

```bash
curl 127.0.0.1:8000
```

## Deploying the Project

Deployed via docker

... missing details about docker

## Running the Data Parser

The parser takes in the folder location of the source data and attempts to convert it to a format that will be easier to work with.

Run it with:

```bash
python3 util/parse-source-data.py [path-of-source-data] --output [location-to-put-it]
```

eg. the example data was generated with (where the source data i downloaded was put outside of the backend folder/repo):

```bash
python3 util/parse-source-data.py ../test_images_labels_targets/test --output data-example
```

Note: you need to pass the folder that contains the subfolders images/ labels/ targets/, the downloaded data has some folder nesting.
