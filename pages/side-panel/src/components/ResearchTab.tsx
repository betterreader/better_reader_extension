import React, { useState } from 'react';
import { ArticleData } from '../SidePanel';

interface ResearchTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
}

interface ResearchResult {
  title: string;
  url: string;
  snippet: string;
}

const ResearchTab: React.FC<ResearchTabProps> = ({ articleData, apiBaseUrl, theme }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ResearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [researchQuery, setResearchQuery] = useState('');
  const [showQueryError, setShowQueryError] = useState(false);

  const handleResearch = async () => {
    if (!researchQuery.trim()) {
      setShowQueryError(true);
      return;
    }
    setShowQueryError(false);
    if (!articleData) {
      setError('No article content available');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/research`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          articleContent: articleData.content,
          articleTitle: articleData.title,
          articleUrl: articleData.url,
          researchQuery: researchQuery.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch research results');
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`h-full flex flex-col p-4 ${theme === 'light' ? 'bg-white' : 'bg-[#1E1E1E]'}`}>
      <div className="mb-4 space-y-3">
        <div>
          <textarea
            value={researchQuery}
            onChange={e => {
              setResearchQuery(e.target.value);
              if (showQueryError) setShowQueryError(false);
            }}
            placeholder="What aspects of this article would you like to research further? (max 200 characters)"
            maxLength={200}
            className={`w-full p-2 rounded-lg border resize-none ${
              theme === 'light'
                ? 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                : 'bg-[#2A2A2A] border-[#444] text-gray-100 placeholder-gray-400'
            } ${showQueryError ? 'border-red-500' : ''}`}
            rows={3}
          />
          {showQueryError && (
            <p className={`mt-1 text-sm ${theme === 'light' ? 'text-red-600' : 'text-red-400'}`}>
              Please enter what you'd like to research
            </p>
          )}
        </div>
        <button
          onClick={handleResearch}
          disabled={isLoading || !articleData}
          className={`w-full py-2 px-4 rounded-lg font-semibold transition-colors ${
            theme === 'light'
              ? 'bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-300'
              : 'bg-[#297FFF] text-white hover:bg-[#1E6FEF] disabled:bg-[#333]'
          }`}>
          {isLoading ? 'Searching...' : 'Find Related Articles'}
        </button>
      </div>

      {error && (
        <div
          className={`p-4 mb-4 rounded-lg ${
            theme === 'light' ? 'bg-red-100 text-red-700' : 'bg-red-900/20 text-red-400'
          }`}>
          {error}
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        {results.map((result, index) => (
          <div key={index} className={`mb-4 p-4 rounded-lg ${theme === 'light' ? 'bg-gray-100' : 'bg-[#2A2A2A]'}`}>
            <span className={`text-sm ${theme === 'light' ? 'text-gray-600' : 'text-gray-400'}`}>
              {new URL(result.url).hostname.replace('www.', '')}
            </span>
            <a
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              className={`text-lg font-semibold mb-2 block hover:underline ${
                theme === 'light' ? 'text-blue-600' : 'text-[#297FFF]'
              }`}>
              {result.title}
            </a>
            <p className={theme === 'light' ? 'text-gray-700' : 'text-gray-300'}>{result.snippet}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResearchTab;
