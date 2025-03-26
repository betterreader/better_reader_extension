-- Create function to search all segments by embedding similarity
CREATE OR REPLACE FUNCTION search_segments(query_embedding vector, match_limit int)
RETURNS TABLE (
  id uuid,
  article_id uuid,
  segment_text text,
  segment_index int,
  created_at timestamp with time zone,
  similarity float
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
    article_segments.created_at,
    1 - (article_segments.embedding <=> query_embedding) as similarity
  FROM
    article_segments
  ORDER BY
    article_segments.embedding <=> query_embedding
  LIMIT match_limit;
END;
$$;

-- Create function to search segments by article ID and embedding similarity
CREATE OR REPLACE FUNCTION search_segments_by_article(query_embedding vector, match_limit int, article_filter uuid)
RETURNS TABLE (
  id uuid,
  article_id uuid,
  segment_text text,
  segment_index int,
  created_at timestamp with time zone,
  similarity float
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
    article_segments.created_at,
    1 - (article_segments.embedding <=> query_embedding) as similarity
  FROM
    article_segments
  WHERE
    article_segments.article_id = article_filter
  ORDER BY
    article_segments.embedding <=> query_embedding
  LIMIT match_limit;
END;
$$;

-- Create function to find similar articles based on embedding similarity
CREATE OR REPLACE FUNCTION find_similar_articles(article_embedding vector, current_article_id uuid, match_limit int)
RETURNS TABLE (
  articles json
) 
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    json_build_object(
      'id', a.id,
      'url', a.url,
      'title', a.title
    ) as articles
  FROM 
    article_segments as s
  JOIN 
    articles as a ON s.article_id = a.id
  WHERE 
    s.article_id != current_article_id
  GROUP BY 
    a.id, a.url, a.title
  ORDER BY 
    MIN(s.embedding <=> article_embedding)
  LIMIT match_limit;
END;
$$;
