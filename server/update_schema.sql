-- Add topics column to articles table
ALTER TABLE articles ADD COLUMN IF NOT EXISTS topics TEXT[];

-- Add keywords and importance_score columns to article_segments table
ALTER TABLE article_segments ADD COLUMN IF NOT EXISTS keywords TEXT[];
ALTER TABLE article_segments ADD COLUMN IF NOT EXISTS importance_score FLOAT DEFAULT 1.0;

-- Create indexes for the new columns
CREATE INDEX IF NOT EXISTS idx_articles_topics ON articles USING GIN (topics);
CREATE INDEX IF NOT EXISTS idx_article_segments_keywords ON article_segments USING GIN (keywords);
