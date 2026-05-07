from __future__ import annotations

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    MetaData,
    Integer,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.types import UserDefinedType


class Geometry(UserDefinedType):
    """Minimal geometry column type for PostGIS without extra dependencies."""

    cache_ok = True

    def __init__(self, geometry_type: str, srid: int = 4326) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw: object) -> str:
        return f"geometry({self.geometry_type},{self.srid})"


metadata = MetaData()


disasters = Table(
    "disasters",
    metadata,
    Column("id", Text, primary_key=True),
    Column("type", Text, nullable=True),
)


image_pairs = Table(
    "image_pairs",
    metadata,
    Column("id", Text, primary_key=True),
    Column(
        "disaster_id",
        Text,
        ForeignKey("disasters.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("pair_id", Text, nullable=False),
    Column("pre_path", Text, nullable=False),
    Column("post_path", Text, nullable=False),
    Column("pre_image_id", Text, nullable=True),
    Column("post_image_id", Text, nullable=True),
    Column("pre_min_lat", Float, nullable=True),
    Column("pre_min_lng", Float, nullable=True),
    Column("pre_max_lat", Float, nullable=True),
    Column("pre_max_lng", Float, nullable=True),
    Column("post_min_lat", Float, nullable=True),
    Column("post_min_lng", Float, nullable=True),
    Column("post_max_lat", Float, nullable=True),
    Column("post_max_lng", Float, nullable=True),
)


locations = Table(
    "locations",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("location_uid", Text, nullable=False),
    Column(
        "image_pair_id",
        Text,
        ForeignKey("image_pairs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("feature_type", Text, nullable=True),
    Column("classification", Text, nullable=True),
    Column("geom", Geometry("Polygon", 4326), nullable=False),
    Column("centroid", Geometry("Point", 4326), nullable=False),
    Column("street", Text, nullable=True),
    Column("city", Text, nullable=True),
    Column("county", Text, nullable=True),
    Column("full_address", Text, nullable=True),
    Column("address_source", Text, nullable=True),
    Column("address_fetched_at", DateTime(timezone=True), nullable=True),
)

Index("ix_locations_geom_gist", locations.c.geom, postgresql_using="gist")
Index("ix_locations_centroid_gist", locations.c.centroid, postgresql_using="gist")
Index("ix_locations_classification", locations.c.classification)


news_articles = Table(
    "news_articles",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "disaster_id",
        Text,
        ForeignKey("disasters.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("source", Text, nullable=True),
    Column("title", Text, nullable=False),
    Column("url", Text, nullable=False),
    Column("published_at", Text, nullable=True),
    Column("summary", Text, nullable=True),
    Column("content", Text, nullable=False),
    UniqueConstraint("url", name="uq_news_articles_url"),
)


news_article_chunks = Table(
    "news_article_chunks",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "article_id",
        BigInteger,
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    UniqueConstraint(
        "article_id",
        "chunk_index",
        name="uq_news_article_chunks_article_id_chunk_index",
    ),
)


@lru_cache(maxsize=4)
def get_engine(database_url: str | None = None) -> Engine:
    db_url = database_url or os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")

    # Pool connections to avoid exhausting max_clients on frequent requests
    return create_engine(
        db_url,
        future=True,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=300,
    )
