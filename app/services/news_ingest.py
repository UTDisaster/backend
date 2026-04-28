from __future__ import annotations

import re
import urllib.request
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlparse

from sqlalchemy import insert, text

from app.db import get_engine, metadata, news_article_chunks, news_articles


URL_RE = re.compile(r"https?://[^\s<>\"]+")


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.paragraphs: list[str] = []
        self._in_title = False
        self._in_paragraph = False
        self._current_paragraph_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "title":
            self._in_title = True
        elif tag == "p":
            self._in_paragraph = True
            self._current_paragraph_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "p":
            self._in_paragraph = False
            paragraph = " ".join(part.strip() for part in self._current_paragraph_parts if part.strip())
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                self.paragraphs.append(paragraph)
            self._current_paragraph_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_paragraph:
            self._current_paragraph_parts.append(data)

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.title_parts)).strip()

    @property
    def content(self) -> str:
        if self.paragraphs:
            return "\n\n".join(self.paragraphs)
        return ""


def extract_urls(text: str) -> list[str]:
    urls = URL_RE.findall(text or "")
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        cleaned = url.rstrip(").,;]")
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def fetch_article(url: str, timeout: int = 20) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        raw = response.read()

    if content_type != "text/html":
        raise ValueError(f"Unsupported content type: {content_type}")

    html = raw.decode("utf-8", errors="ignore")
    parser = _ArticleParser()
    parser.feed(html)

    title = parser.title or urlparse(url).netloc
    content = parser.content.strip()
    if not content:
        content = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()

    content = content[:20000]
    return {
        "title": title,
        "source": urlparse(url).netloc,
        "url": url,
        "content": content,
    }


def fetch_articles(urls: Iterable[str]) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    for url in urls:
        articles.append(fetch_article(url))
    return articles


def store_articles(articles: Iterable[dict[str, str]], disaster_id: str | None = None) -> list[dict[str, str]]:
    engine = get_engine()
    stored: list[dict[str, str]] = []

    with engine.begin() as conn:
        metadata.create_all(conn)

        for article in articles:
            url = article.get("url")
            if not url:
                continue

            existing = conn.execute(
                text("SELECT id FROM news_articles WHERE url = :url"),
                {"url": url},
            ).scalar_one_or_none()

            if existing:
                conn.execute(text("DELETE FROM news_articles WHERE id = :id"), {"id": existing})

            article_values = {
                "disaster_id": disaster_id or article.get("disaster_id"),
                "source": article.get("source"),
                "title": article.get("title") or urlparse(url).netloc,
                "url": url,
                "published_at": article.get("published_at"),
                "summary": article.get("summary"),
                "content": article.get("content", ""),
            }

            article_id = conn.execute(
                insert(news_articles).returning(news_articles.c.id),
                [article_values],
            ).scalar_one()

            content = article_values["content"]
            chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()] or [content[:1400]]
            chunk_rows = [
                {"article_id": article_id, "chunk_index": idx, "content": chunk[:1400]}
                for idx, chunk in enumerate(chunks)
            ]
            if chunk_rows:
                conn.execute(insert(news_article_chunks), chunk_rows)

            stored.append(
                {
                    "id": str(article_id),
                    "title": article_values["title"],
                    "url": url,
                    "disaster_id": article_values["disaster_id"] or "",
                }
            )

    return stored
