from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.news_ingest import extract_urls, fetch_articles, store_articles


router = APIRouter(prefix="/news", tags=["news"])


class NewsIngestRequest(BaseModel):
    text: str | None = None
    urls: list[str] = Field(default_factory=list)
    disaster_id: str | None = None


@router.post("/ingest")
def ingest_news(req: NewsIngestRequest) -> dict[str, Any]:
    urls: list[str] = []
    seen: set[str] = set()

    for url in req.urls:
        if url not in seen:
            seen.add(url)
            urls.append(url)

    if req.text:
        for url in extract_urls(req.text):
            if url not in seen:
                seen.add(url)
                urls.append(url)

    if not urls:
        raise HTTPException(
            status_code=400,
            detail="Provide urls or text containing article URLs.",
        )

    articles = fetch_articles(urls)
    stored = store_articles(articles, disaster_id=req.disaster_id)

    return {
        "extracted_urls": urls,
        "stored": stored,
        "stored_count": len(stored),
    }
