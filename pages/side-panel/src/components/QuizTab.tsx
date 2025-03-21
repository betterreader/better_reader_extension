// src/components/QuizTab.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ArticleData } from '../SidePanel';

interface QuizTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
}

interface QuizQuestion {
  question: string;
  options: string[];
  answer: number; // correct option index
  explanation?: string;
}

interface QuizData {
  questions: QuizQuestion[];
}

interface Message {
  sender: 'bot' | 'user';
  text: string;
}

const QuizTab: React.FC<QuizTabProps> = ({ articleData, apiBaseUrl, theme }) => {
  const [quizMessages, setQuizMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'I can generate quiz questions about this article. Try clicking "Generate Quiz" below!',
    },
  ]);
  const [currentQuizData, setCurrentQuizData] = useState<QuizData | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');
  const [showCustomPrompt, setShowCustomPrompt] = useState(false);
  const quizContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (quizContainerRef.current) {
      quizContainerRef.current.scrollTop = quizContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [quizMessages]);

  useEffect(() => {
    if (currentQuizData) {
      setTimeout(scrollToBottom, 100); // Add a small delay to ensure content is rendered
    }
  }, [currentQuizData]);

  const generateQuiz = (prompt: string = '') => {
    // Append a loading message
    setQuizMessages(prev => [...prev, { sender: 'bot', text: 'Generating quiz questions...' }]);

    if (!articleData || !articleData.content) {
      setQuizMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1] = {
          sender: 'bot',
          text: "I couldn't extract any content from this page. Please try a different article.",
        };
        return newMsgs;
      });
      return;
    }

    const timestamp = new Date().getTime();
    const requestData: any = {
      articleContent: articleData.content,
      articleTitle: articleData.title,
      articleUrl: articleData.url,
      timestamp: timestamp,
    };

    if (prompt) {
      requestData.customPrompt = prompt;
    }

    fetch(`${apiBaseUrl}/api/generate-quiz`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Network response was not ok: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Remove loading message
        setQuizMessages(prev => prev.slice(0, -1));
        if (data.error) {
          setQuizMessages(prev => [...prev, { sender: 'bot', text: `Error: ${data.error}` }]);
          return;
        }
        setCurrentQuizData(data);
        displayQuiz(data, prompt);
      })
      .catch(error => {
        console.error('Error generating quiz:', error);
        setQuizMessages(prev => {
          const newMsgs = [...prev];
          newMsgs[newMsgs.length - 1] = { sender: 'bot', text: `Error: ${error.message}` };
          return newMsgs;
        });
      });
  };

  const displayQuiz = (quizData: QuizData, prompt: string) => {
    const introText = prompt
      ? `Here are some quiz questions based on your request: "${prompt}"`
      : 'Here are some quiz questions based on the article:';
    setQuizMessages(prev => [...prev, { sender: 'bot', text: introText }]);
  };

  const handleOptionSelection = (questionIndex: number, optionIndex: number) => {
    if (!currentQuizData) return;
    const question = currentQuizData.questions[questionIndex];
    let feedback = '';
    if (optionIndex === question.answer) {
      feedback = 'Correct!';
    } else {
      feedback = `Incorrect. The correct answer is: ${question.options[question.answer]}`;
    }
    // Append feedback message
    setQuizMessages(prev => [...prev, { sender: 'bot', text: feedback }]);
  };

  return (
    <div className="flex flex-col h-full">
      <div ref={quizContainerRef} className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 max-h-[calc(100vh-160px)]">
        {quizMessages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg break-words leading-6 shadow-sm ${
              theme === 'light' ? 'bg-gray-100 text-gray-800' : 'bg-[#252526] text-[#D4D4D4]'
            }`}>
            {msg.text}
          </div>
        ))}
        {currentQuizData &&
          currentQuizData.questions.map((question, qIndex) => (
            <div
              key={qIndex}
              className={`p-4 rounded-lg shadow-md border ${
                theme === 'light' ? 'bg-white border-gray-200' : 'bg-[#2D2D2D] border-[#404040]'
              }`}
              data-question-index={qIndex}>
              <div className={`mb-3 text-lg font-semibold ${theme === 'light' ? 'text-gray-900' : 'text-white'}`}>
                {qIndex + 1}. {question.question}
              </div>
              <div className="flex flex-col gap-2">
                {question.options.map((option, oIndex) => (
                  <div
                    key={oIndex}
                    className={`p-3 border rounded cursor-pointer transition-colors ${
                      theme === 'light'
                        ? 'border-gray-200 text-gray-700 hover:bg-gray-50'
                        : 'border-[#404040] text-[#E0E0E0] hover:bg-[#404040]'
                    }`}
                    onClick={() => handleOptionSelection(qIndex, oIndex)}>
                    {option}
                  </div>
                ))}
              </div>
              {question.explanation && (
                <div
                  className={`mt-3 p-3 border-l-4 border-[#0078D4] rounded text-sm ${
                    theme === 'light' ? 'bg-blue-50 text-gray-700' : 'bg-[#1E1E1E] text-[#E0E0E0]'
                  }`}>
                  {question.explanation}
                </div>
              )}
            </div>
          ))}
      </div>
      <div className={`p-3 border-t ${theme === 'light' ? 'border-gray-200 bg-white' : 'border-[#333] bg-[#1E1E1E]'}`}>
        <div className="flex gap-2 mb-2">
          <button
            className="bg-[#0078D4] text-white py-2 px-4 rounded hover:bg-[#005A9E] transition-colors"
            onClick={() => generateQuiz()}>
            Generate Quiz
          </button>
          <button
            className={`py-2 px-4 rounded transition-colors ${
              theme === 'light'
                ? 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50'
                : 'bg-transparent border border-[#333] text-[#D4D4D4] hover:bg-white/10'
            }`}
            onClick={() => setShowCustomPrompt(prev => !prev)}>
            Custom Quiz
          </button>
        </div>
        {showCustomPrompt && (
          <div className="flex gap-2">
            <input
              type="text"
              className={`flex-1 p-2 border rounded outline-none ${
                theme === 'light'
                  ? 'border-gray-300 bg-white text-gray-900 focus:border-blue-500'
                  : 'border-[#3C3C3C] bg-[#252526] text-[#D4D4D4] focus:border-[#0078D4]'
              }`}
              placeholder="Enter a custom prompt for quiz generation"
              value={customPrompt}
              onChange={e => setCustomPrompt(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  generateQuiz(customPrompt);
                  setShowCustomPrompt(false);
                  setCustomPrompt('');
                }
              }}
            />
            <button
              className="bg-[#0078D4] text-white py-2 px-4 rounded hover:bg-[#005A9E] transition-colors"
              onClick={() => {
                generateQuiz(customPrompt);
                setShowCustomPrompt(false);
                setCustomPrompt('');
              }}>
              Generate
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuizTab;
