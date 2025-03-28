// src/components/QuizTab.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ArticleData, QuizData, QuizQuestion, QuestionState, Message } from '../SidePanel';

interface QuizTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
  quizMessages: Message[];
  setQuizMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  currentQuizData: QuizData | null;
  setCurrentQuizData: React.Dispatch<React.SetStateAction<QuizData | null>>;
  questionStates: QuestionState[];
  setQuestionStates: React.Dispatch<React.SetStateAction<QuestionState[]>>;
}

const QuizTab: React.FC<QuizTabProps> = ({
  articleData,
  apiBaseUrl,
  theme,
  quizMessages,
  setQuizMessages,
  currentQuizData,
  setCurrentQuizData,
  questionStates,
  setQuestionStates,
}) => {
  const [customPrompt, setCustomPrompt] = useState('');
  const [showCustomPrompt, setShowCustomPrompt] = useState(false);
  const [showLevelDialog, setShowLevelDialog] = useState(false);
  const [selectedLevel, setSelectedLevel] = useState<string>('');
  const [pendingPrompt, setPendingPrompt] = useState<string>('');
  const [shouldScroll, setShouldScroll] = useState(false);
  const quizContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (quizContainerRef.current) {
      quizContainerRef.current.scrollTop = quizContainerRef.current.scrollHeight;
    }
  };

  // Initialize question states when quiz data changes
  useEffect(() => {
    if (currentQuizData) {
      // Initialize state for each question
      setQuestionStates(
        currentQuizData.questions.map(() => ({
          answered: false,
          selectedOption: null,
          isCorrect: null,
        })),
      );
    }
  }, [currentQuizData, setQuestionStates]);

  const generateQuiz = (prompt: string = '', level: string = '') => {
    // Append a loading message
    setQuizMessages(prev => [...prev, { sender: 'bot', text: 'Generating quiz questions...' }]);
    setShouldScroll(true); // Request scroll when quiz is generated

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
      randomSeed: Math.random(), // add randomness to the request
    };

    if (prompt) {
      requestData.customPrompt = prompt;
    }

    if (level) {
      requestData.userLevel = level;
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

        // Handle different API response formats (correctAnswer vs answer)
        if (data.questions && data.questions.length > 0) {
          // Map the API response to ensure consistent property names
          const formattedData = {
            questions: data.questions.map((q: QuizQuestion) => ({
              ...q,
              // Ensure we always have an 'answer' property
              answer: q.correctAnswer !== undefined ? q.correctAnswer : q.answer,
            })),
          };
          setCurrentQuizData(formattedData);
          displayQuiz(formattedData, prompt, level);
        } else {
          setQuizMessages(prev => [...prev, { sender: 'bot', text: 'Error: Invalid quiz data format' }]);
        }
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

  const displayQuiz = (quizData: QuizData, prompt: string, level: string = '') => {};

  const handleOptionSelection = (questionIndex: number, optionIndex: number) => {
    if (!currentQuizData) return;

    // If question is already answered, do nothing
    if (questionStates[questionIndex].answered) return;

    const question = currentQuizData.questions[questionIndex];
    // Use answer property (which is now guaranteed to exist)
    const correctAnswerIndex = question.answer;
    const isCorrect = optionIndex === correctAnswerIndex;

    // Update question state
    setQuestionStates(prev => {
      const newStates = [...prev];
      newStates[questionIndex] = {
        answered: true,
        selectedOption: optionIndex,
        isCorrect: isCorrect,
      };
      return newStates;
    });
  };

  const handleGenerateQuizClick = (customPromptValue: string = '') => {
    setPendingPrompt(customPromptValue);
    setShowLevelDialog(true);
  };

  const handleLevelSelection = (level: string) => {
    setSelectedLevel(level);
    setShowLevelDialog(false);
    generateQuiz(pendingPrompt, level);

    if (pendingPrompt) {
      setShowCustomPrompt(false);
      setCustomPrompt('');
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div ref={quizContainerRef} className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 max-h-[calc(100vh-220px)]">
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
                      questionStates[qIndex]?.answered && questionStates[qIndex]?.selectedOption === oIndex
                        ? questionStates[qIndex]?.isCorrect
                          ? theme === 'light'
                            ? 'bg-green-100 border-green-500 text-green-800'
                            : 'bg-green-900 border-green-500 text-green-100'
                          : theme === 'light'
                            ? 'bg-red-100 border-red-500 text-red-800'
                            : 'bg-red-900 border-red-500 text-red-100'
                        : questionStates[qIndex]?.answered && oIndex === question.answer
                          ? theme === 'light'
                            ? 'bg-green-100 border-green-500 text-green-800'
                            : 'bg-green-900 border-green-500 text-green-100'
                          : theme === 'light'
                            ? 'border-gray-200 text-gray-700 hover:bg-gray-50'
                            : 'border-[#404040] text-[#E0E0E0] hover:bg-[#404040]'
                    }`}
                    onClick={() => handleOptionSelection(qIndex, oIndex)}>
                    {option}
                  </div>
                ))}
              </div>
              {questionStates[qIndex]?.answered && (
                <div
                  className={`mt-3 p-3 border-l-4 rounded text-sm ${
                    questionStates[qIndex]?.isCorrect
                      ? theme === 'light'
                        ? 'border-green-500 bg-green-50 text-green-800'
                        : 'border-green-500 bg-green-900/20 text-green-100'
                      : theme === 'light'
                        ? 'border-red-500 bg-red-50 text-red-800'
                        : 'border-red-500 bg-red-900/20 text-red-100'
                  }`}>
                  {questionStates[qIndex]?.isCorrect
                    ? 'Correct!'
                    : `Incorrect. The correct answer is: ${question.options[question.answer || 0]}`}
                  {question.explanation && (
                    <div className="mt-2">
                      <strong>Explanation:</strong> {question.explanation}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
      </div>

      {/* Level Selection Dialog */}
      {showLevelDialog && (
        <div className={`fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50`}>
          <div
            className={`p-6 rounded-lg shadow-lg max-w-md w-full ${theme === 'light' ? 'bg-white' : 'bg-[#2D2D2D]'}`}>
            <h3 className={`text-lg font-semibold mb-4 ${theme === 'light' ? 'text-gray-900' : 'text-white'}`}>
              Select your level of understanding
            </h3>
            <p className={`mb-4 ${theme === 'light' ? 'text-gray-700' : 'text-gray-300'}`}>
              This helps tailor the quiz questions to your knowledge level.
            </p>
            <div className="flex flex-col gap-3">
              <button
                className={`p-3 border rounded text-left transition-colors ${
                  theme === 'light'
                    ? 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50'
                    : 'bg-transparent border border-[#333] text-[#D4D4D4] hover:bg-white/10'
                }`}
                onClick={() => handleLevelSelection('beginner')}>
                <div className={`font-medium ${theme === 'light' ? 'text-gray-900' : 'text-white'}`}>Beginner</div>
                <div className={`text-sm ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
                  I'm new to this topic and need basic questions
                </div>
              </button>
              <button
                className={`p-3 border rounded text-left transition-colors ${
                  theme === 'light'
                    ? 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50'
                    : 'bg-transparent border border-[#333] text-[#D4D4D4] hover:bg-white/10'
                }`}
                onClick={() => handleLevelSelection('intermediate')}>
                <div className={`font-medium ${theme === 'light' ? 'text-gray-900' : 'text-white'}`}>Intermediate</div>
                <div className={`text-sm ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
                  I have some knowledge and want moderate difficulty
                </div>
              </button>
              <button
                className={`p-3 border rounded text-left transition-colors ${
                  theme === 'light'
                    ? 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50'
                    : 'bg-transparent border border-[#333] text-[#D4D4D4] hover:bg-white/10'
                }`}
                onClick={() => handleLevelSelection('expert')}>
                <div className={`font-medium ${theme === 'light' ? 'text-gray-900' : 'text-white'}`}>Expert</div>
                <div className={`text-sm ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
                  I'm well-versed in this topic and want challenging questions
                </div>
              </button>
            </div>
            <div className="mt-4 flex justify-end">
              <button
                className={`py-2 px-4 rounded ${
                  theme === 'light'
                    ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    : 'bg-[#333] text-[#D4D4D4] hover:bg-[#444]'
                }`}
                onClick={() => setShowLevelDialog(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div
        className={`p-3 border-t sticky bottom-0 ${theme === 'light' ? 'border-gray-200 bg-white' : 'border-[#333] bg-[#1E1E1E]'}`}>
        <div className="flex gap-2 mb-2">
          <button
            className="bg-[#0078D4] text-white py-2 px-4 rounded hover:bg-[#005A9E] transition-colors"
            onClick={() => handleGenerateQuizClick()}>
            Generate Quiz
          </button>
          {/* <button
            className={`py-2 px-4 rounded transition-colors ${
              theme === 'light'
                ? 'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50'
                : 'bg-transparent border border-[#333] text-[#D4D4D4] hover:bg-white/10'
            }`}
            onClick={() => setShowCustomPrompt(prev => !prev)}>
            Custom Quiz
          </button> */}
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
                  handleGenerateQuizClick(customPrompt);
                }
              }}
            />
            <button
              className="bg-[#0078D4] text-white py-2 px-4 rounded hover:bg-[#005A9E] transition-colors"
              onClick={() => {
                handleGenerateQuizClick(customPrompt);
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
