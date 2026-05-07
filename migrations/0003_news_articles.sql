CREATE TABLE IF NOT EXISTS news_articles (
    id bigserial PRIMARY KEY,
    disaster_id text REFERENCES disasters(id) ON DELETE SET NULL,
    source text,
    title text NOT NULL,
    url text NOT NULL UNIQUE,
    published_at text,
    summary text,
    content text NOT NULL
);

CREATE TABLE IF NOT EXISTS news_article_chunks (
    id bigserial PRIMARY KEY,
    article_id bigint NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    CONSTRAINT uq_news_article_chunks_article_id_chunk_index
        UNIQUE (article_id, chunk_index)
);
