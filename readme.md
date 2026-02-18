# Backend

## Info

## Setup

Recommended to use [https://docs.astral.sh/uv/getting-started/installation/](uv) to manage virtual environment and pip packages.

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

...