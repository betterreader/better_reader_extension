-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Articles table - stores metadata about processed articles
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    topics TEXT[], -- Array of topics/keywords for the article
    CONSTRAINT unique_article_url UNIQUE (url)
);

-- Article segments table - stores text segments and their vector embeddings
CREATE TABLE IF NOT EXISTS article_segments (
    id UUID PRIMARY KEY,
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    segment_text TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI's text-embedding-3-small has 1536 dimensions
    segment_index INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    keywords TEXT[], -- Keywords specific to this segment
    importance_score FLOAT DEFAULT 1.0, -- Score indicating the importance of this segment
    CONSTRAINT unique_segment_per_article UNIQUE (article_id, segment_index)
);

-- Create a vector index for faster similarity searches
CREATE INDEX IF NOT EXISTS article_segments_embedding_idx ON article_segments 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Add indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_article_segments_article_id ON article_segments (article_id);
CREATE INDEX IF NOT EXISTS idx_articles_user_id ON articles (user_id);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url);
CREATE INDEX IF NOT EXISTS idx_articles_topics ON articles USING GIN (topics);
CREATE INDEX IF NOT EXISTS idx_article_segments_keywords ON article_segments USING GIN (keywords);
