'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';

interface ShortResponseProps {
  questions: Array<{
    question: string;
    id: string;
  }>;
  onSubmitResponse: (questionId: string, response: string) => Promise<string>;
}

export default function ShortResponse({ questions, onSubmitResponse }: ShortResponseProps) {
  const [responses, setResponses] = useState<Record<string, string>>({});
  const [feedback, setFeedback] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  const handleResponseChange = (questionId: string, value: string) => {
    setResponses(prev => ({
      ...prev,
      [questionId]: value,
    }));
  };

  const handleSubmit = async (questionId: string) => {
    if (!responses[questionId]?.trim()) return;

    setLoading(prev => ({ ...prev, [questionId]: true }));
    try {
      const feedbackResponse = await onSubmitResponse(questionId, responses[questionId]);
      setFeedback(prev => ({
        ...prev,
        [questionId]: feedbackResponse,
      }));
    } catch (error) {
      console.error('Error getting feedback:', error);
      setFeedback(prev => ({
        ...prev,
        [questionId]: 'Failed to get feedback. Please try again.',
      }));
    } finally {
      setLoading(prev => ({ ...prev, [questionId]: false }));
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Reflection Questions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {questions.map(question => (
          <div key={question.id} className="space-y-4">
            <div>
              <Label htmlFor={question.id}>{question.question}</Label>
              <Textarea
                id={question.id}
                value={responses[question.id] || ''}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                  handleResponseChange(question.id, e.target.value)
                }
                className="mt-2"
                rows={4}
              />
            </div>
            <Button
              onClick={() => handleSubmit(question.id)}
              disabled={loading[question.id] || !responses[question.id]?.trim()}>
              {loading[question.id] ? 'Getting Feedback...' : 'Get Feedback'}
            </Button>
            {feedback[question.id] && (
              <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <h4 className="font-medium mb-2">AI Feedback:</h4>
                <p className="text-gray-700 dark:text-gray-300">{feedback[question.id]}</p>
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
