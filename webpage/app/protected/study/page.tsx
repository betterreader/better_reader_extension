'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { createClient } from '@/utils/supabase/client';
import ArticleSummary from '@/components/article-summary';
import ArticleContext from '@/components/article-context';
import QuizQuestions from '@/components/quiz-questions';
import ShortResponse from '@/components/short-response';
import FutureResearch from '@/components/future-research';
import { AnalysisResult } from '@/types/analysis';

export default function StudyPageWrapper() {
  return (
    <Suspense fallback={<Loading />}>
      <StudyPage />
    </Suspense>
  );
}

function StudyPage() {
  const searchParams = useSearchParams();
  const analysisId = searchParams.get('id');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      if (!analysisId) {
        setLoading(false);
        return;
      }

      try {
        const supabase = createClient();
        const { data, error } = await supabase.from('analysis').select('*').eq('id', Number(analysisId)).single();

        if (error) throw error;

        if (data) {
          // Transform snake_case to camelCase
          setResult({
            title: data.title,
            summary: data.summary,
            context: data.context,
            keyTerms: data.key_terms as { term: string; definition: string }[],
            questions: data.questions as {
              question: string;
              options: string[];
              correctAnswer: number;
            }[],
            shortResponseQuestions: data.short_response_questions as {
              id: string;
              question: string;
            }[],
            futureResearch: data.future_research as {
              topic: string;
              description: string;
            }[],
            articleContent: data.article_content,
            analysisId: data.id,
          });
        }
      } catch (err) {
        console.error('Error fetching analysis:', err);
        setError('Failed to load analysis data');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [analysisId]);

  const handleShortResponseSubmit = async (questionId: string, response: string) => {
    if (!result) return;

    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          response,
          articleContent: result.articleContent,
        }),
      });

      if (!res.ok) {
        throw new Error('Failed to get feedback');
      }

      const data = await res.json();
      return data.feedback;
    } catch (error) {
      console.error('Error getting feedback:', error);
      throw error;
    }
  };

  if (loading) return <Loading />;
  if (error) return <ErrorMessage error={error} />;
  if (!result) return <NoDataMessage />;

  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center">Study Session</h1>

      <div className="space-y-8">
        <h1 className="text-4xl font-bold mb-8 text-center">{result.title}</h1>
        <ArticleSummary summary={result.summary} />
        <ArticleContext context={result.context} keyTerms={result.keyTerms} />
        <QuizQuestions questions={result.questions} />
        <ShortResponse questions={result.shortResponseQuestions} onSubmitResponse={handleShortResponseSubmit} />
        <FutureResearch recommendations={result.futureResearch} />
      </div>
    </main>
  );
}

function Loading() {
  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center">Study Session</h1>
      <p className="text-center">Loading analysis data...</p>
    </main>
  );
}

function ErrorMessage({ error }: { error: string }) {
  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center">Study Session</h1>
      <p className="text-center text-red-500">{error}</p>
    </main>
  );
}

function NoDataMessage() {
  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center">Study Session</h1>
      <p className="text-center">No analysis data available. Please analyze an article first.</p>
    </main>
  );
}
