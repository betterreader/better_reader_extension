'use client';

import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { RadioGroup, RadioGroupItem } from './ui/radio-group';
import { Label } from './ui/label';
import { Button } from './ui/button';

interface Question {
  question: string;
  options: string[];
  correctAnswer: number;
}

interface QuizQuestionsProps {
  questions: Question[];
}

export default function QuizQuestions({ questions }: QuizQuestionsProps) {
  if (!questions) {
    console.log('No questions found');
    return null;
  }

  // Initialize with an empty string for each question.
  const [selectedAnswers, setSelectedAnswers] = useState<string[]>(new Array(questions.length).fill(''));
  const [showResults, setShowResults] = useState(false);

  const handleAnswerSelect = (questionIndex: number, optionValue: string) => {
    setSelectedAnswers(prev => {
      const newAnswers = [...prev];
      newAnswers[questionIndex] = optionValue;
      return newAnswers;
    });
  };

  const handleSubmit = () => {
    setShowResults(true);
  };

  const getScore = () => {
    return questions.reduce((score, question, index) => {
      // Convert the stored string to a number for comparison.
      return score + (parseInt(selectedAnswers[index]) === question.correctAnswer ? 1 : 0);
    }, 0);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Comprehension Questions</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {questions.map((question, questionIndex) => (
          <div key={questionIndex} className="space-y-4">
            <p className="font-medium">
              {questionIndex + 1}. {question.question}
            </p>
            <RadioGroup
              value={selectedAnswers[questionIndex]}
              onValueChange={value => handleAnswerSelect(questionIndex, value)}
              className="space-y-2">
              {question.options.map((option, optionIndex) => (
                <div key={optionIndex} className="flex items-center space-x-2">
                  <RadioGroupItem
                    value={optionIndex.toString()}
                    id={`q${questionIndex}-o${optionIndex}`}
                    disabled={showResults}
                  />
                  <Label
                    htmlFor={`q${questionIndex}-o${optionIndex}`}
                    className={
                      showResults
                        ? optionIndex === question.correctAnswer
                          ? 'text-green-600 dark:text-green-400'
                          : selectedAnswers[questionIndex] === optionIndex.toString()
                            ? 'text-red-600 dark:text-red-400'
                            : ''
                        : ''
                    }>
                    {option}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        ))}

        {!showResults && (
          <Button
            onClick={handleSubmit}
            disabled={selectedAnswers.some(answer => answer === '')}
            className="w-full mt-4">
            Submit Answers
          </Button>
        )}

        {showResults && (
          <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
            <p className="text-lg font-medium">
              Your Score: {getScore()} out of {questions.length}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
