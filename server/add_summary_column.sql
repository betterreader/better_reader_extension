-- Add summary column to articles table
ALTER TABLE articles ADD COLUMN IF NOT EXISTS summary TEXT;

-- Create an index on the summary column for faster text searches
CREATE INDEX IF NOT EXISTS idx_articles_summary ON articles USING GIN (to_tsvector('english', summary));
