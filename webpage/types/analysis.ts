export interface AnalysisResult {
  title: string;
  summary: string;
  context: string;
  keyTerms: Array<{
    term: string;
    definition: string;
  }>;
  questions: Array<{
    question: string;
    options: string[];
    correctAnswer: number;
  }>;
  shortResponseQuestions: Array<{
    id: string;
    question: string;
  }>;
  futureResearch: Array<{
    topic: string;
    description: string;
  }>;
  articleContent: string;
  analysisId: number;
}
