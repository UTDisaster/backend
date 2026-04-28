#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import insert, text

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import get_engine, metadata, news_article_chunks, news_articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import news articles into the database.")
    parser.add_argument("input_json", help="Path to a JSON file containing article objects.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="Optional DATABASE_URL override for the import step",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear existing article rows before importing.",
    )
    return parser.parse_args()


def _load_articles(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "articles" in payload:
        payload = payload["articles"]

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of articles or an object with an 'articles' key")

    articles: list[dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Article at index {idx} is not an object")

        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title or not content:
            raise ValueError(f"Article at index {idx} must include non-empty 'title' and 'content'")

        articles.append(item)

    return articles


def _split_chunks(content: str, max_len: int = 1400) -> list[str]:
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return [content[:max_len]] if content else []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n\n" + paragraph
        if len(candidate) <= max_len:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_len:
            current = paragraph
        else:
            for start in range(0, len(paragraph), max_len):
                chunks.append(paragraph[start : start + max_len])

    if current:
        chunks.append(current)

    return chunks


def run_import_step(input_json: Path, db_url: str | None = None, truncate: bool = False) -> dict[str, int]:
    articles = _load_articles(input_json)
    engine = get_engine(db_url)

    with engine.begin() as conn:
        metadata.create_all(conn)

        if truncate:
            conn.execute(text("TRUNCATE TABLE news_article_chunks, news_articles RESTART IDENTITY CASCADE"))

        article_rows: list[dict[str, Any]] = []
        chunk_rows: list[dict[str, Any]] = []

        for article in articles:
            article_rows.append(
                {
                    "disaster_id": article.get("disaster_id"),
                    "source": article.get("source"),
                    "title": str(article.get("title")).strip(),
                    "url": article.get("url"),
                    "published_at": article.get("published_at"),
                    "summary": article.get("summary"),
                    "content": str(article.get("content")).strip(),
                }
            )

        if article_rows:
            result = conn.execute(insert(news_articles).returning(news_articles.c.id), article_rows)
            article_ids = result.scalars().all()

            for article_id, article in zip(article_ids, articles):
                chunks = _split_chunks(str(article.get("content") or "").strip())
                for chunk_index, chunk in enumerate(chunks):
                    chunk_rows.append(
                        {
                            "article_id": article_id,
                            "chunk_index": chunk_index,
                            "content": chunk,
                        }
                    )

        if chunk_rows:
            conn.execute(insert(news_article_chunks), chunk_rows)

    return {"articles": len(article_rows), "chunks": len(chunk_rows)}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    result = run_import_step(input_path, db_url=args.db_url, truncate=args.truncate)
    print(f"Imported articles: {result['articles']}")
    print(f"Imported chunks: {result['chunks']}")


if __name__ == "__main__":
    main()
