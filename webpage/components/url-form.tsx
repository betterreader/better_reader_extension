'use client';

import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { AnalysisResult } from '@/types/analysis';

export default function UrlForm({ onSubmit }: { onSubmit: (data: AnalysisResult) => void }) {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze article');
      }

      const data = await response.json();
      onSubmit({
        title: data.title,
        summary: data.summary,
        context: data.context,
        keyTerms: data.keyTerms,
        questions: data.questions,
        shortResponseQuestions: data.shortResponseQuestions,
        futureResearch: data.futureResearch,
        articleContent: data.articleContent,
        analysisId: data.analysisId,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="url">Article URL</Label>
        <Input
          id="url"
          type="url"
          placeholder="https://example.com/article"
          value={url}
          onChange={e => setUrl(e.target.value)}
          required
          className="w-full"
        />
      </div>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button type="submit" disabled={loading} className="w-full">
        {loading ? 'Analyzing...' : 'Analyze Article'}
      </Button>
    </form>
  );
}
