-- Create or replace function to match segments across all articles with enhanced metadata
CREATE OR REPLACE FUNCTION match_segments(
    query_embedding VECTOR(1536),
    match_threshold FLOAT,
    match_limit INT
)
RETURNS TABLE (
    id UUID,
    article_id UUID,
    segment_text TEXT,
    segment_index INT,
    similarity FLOAT,
    keywords TEXT[],
    importance_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        article_segments.id,
        article_segments.article_id,
        article_segments.segment_text,
        article_segments.segment_index,
        1 - (article_segments.embedding <=> query_embedding) AS similarity,
        article_segments.keywords,
        article_segments.importance_score
    FROM
        article_segments
    WHERE
        1 - (article_segments.embedding <=> query_embedding) > match_threshold
    ORDER BY
        article_segments.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$;

-- Create or replace function to match segments within a specific article with enhanced metadata
CREATE OR REPLACE FUNCTION match_article_segments(
    query_embedding VECTOR(1536),
    article_filter UUID,
    match_threshold FLOAT,
    match_limit INT
)
RETURNS TABLE (
    id UUID,
    article_id UUID,
    segment_text TEXT,
    segment_index INT,
    similarity FLOAT,
    keywords TEXT[],
    importance_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        article_segments.id,
        article_segments.article_id,
        article_segments.segment_text,
        article_segments.segment_index,
        1 - (article_segments.embedding <=> query_embedding) AS similarity,
        article_segments.keywords,
        article_segments.importance_score
    FROM
        article_segments
    WHERE
        article_segments.article_id = article_filter
        AND 1 - (article_segments.embedding <=> query_embedding) > match_threshold
    ORDER BY
        article_segments.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$;

-- Create or replace function to find similar articles with enhanced metadata
CREATE OR REPLACE FUNCTION find_similar_articles(
    article_embedding VECTOR(1536),
    current_article_id UUID,
    match_limit INT
)
RETURNS TABLE (
    similarity FLOAT,
    articles JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        1 - (article_segments.embedding <=> article_embedding) AS similarity,
        jsonb_build_object(
            'id', a.id,
            'url', a.url,
            'title', a.title,
            'topics', a.topics
        ) AS articles
    FROM
        article_segments
    JOIN
        articles a ON article_segments.article_id = a.id
    WHERE
        article_segments.article_id != current_article_id
    ORDER BY
        article_segments.embedding <=> article_embedding
    LIMIT match_limit;
END;
$$;
