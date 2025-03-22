'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import UrlForm from '@/components/url-form';
import { AnalysisResult } from '@/types/analysis';

export default function AnalyzePage() {
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleAnalysisComplete = (data: AnalysisResult) => {
    // Redirect to study page with just the analysis ID
    router.push(`/protected/study?id=${data.analysisId}`);
  };

  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center">Analyze Article</h1>

      <div className="space-y-8">
        <div className="max-w-xl mx-auto">
          <UrlForm onSubmit={handleAnalysisComplete} />
        </div>
      </div>
    </main>
  );
}
