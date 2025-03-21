import React, { useState, useEffect } from 'react';
import ChatTab from '@src/components/ChatTab';
import QuizTab from '@src/components/QuizTab';
import { ToggleButton } from '@extension/ui';
import { exampleThemeStorage } from '@extension/storage';
import { useStorage } from '@extension/shared';

export interface ArticleData {
  content: string;
  title: string;
  url: string;
}

// Define quiz interfaces here so they can be shared
export interface QuizQuestion {
  question: string;
  options: string[];
  answer?: number; // correct option index (old format)
  correctAnswer?: number; // correct option index (new format)
  explanation?: string;
}

export interface QuizData {
  questions: QuizQuestion[];
}

export interface QuestionState {
  answered: boolean;
  selectedOption: number | null;
  isCorrect: boolean | null;
}

export interface Message {
  sender: 'bot' | 'user';
  text: string;
}

const API_BASE_URL = 'http://localhost:5007';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'quiz'>('chat');
  const [articleData, setArticleData] = useState<ArticleData | null>(null);
  const theme = useStorage(exampleThemeStorage);

  // Lift quiz state to parent component
  const [quizMessages, setQuizMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'I can generate quiz questions about this article. Try clicking "Generate Quiz" below!',
    },
  ]);
  const [currentQuizData, setCurrentQuizData] = useState<QuizData | null>(null);
  const [questionStates, setQuestionStates] = useState<QuestionState[]>([]);

  useEffect(() => {
    // Fetch article content from background script on mount
    chrome.runtime.sendMessage({ action: 'getArticleContent' }, (response: any) => {
      if (response && response.content) {
        setArticleData({
          content: response.content,
          title: response.title,
          url: response.url,
        });
      } else {
        console.error('Failed to get article content:', response?.error);
      }
    });
  }, []);

  return (
    // Changed fixed width and height to fluid classes
    <div
      className={`w-full h-screen rounded-t-lg overflow-hidden flex flex-col ${
        theme === 'light' ? 'bg-white' : 'bg-[#1E1E1E]'
      }`}>
      <header
        className={`text-2xl font-extrabold p-6 flex justify-between items-center ${
          theme === 'light' ? 'bg-gray-100 text-gray-900' : 'bg-[#1A1A1A] text-white'
        }`}>
        <span>BetterReader</span>
        {/*TODO: change this text into an icon*/}
        <ToggleButton>Toggle Mode</ToggleButton>
      </header>
      <div
        className={`flex border-b ${theme === 'light' ? 'bg-gray-100 border-gray-200' : 'bg-[#1A1A1A] border-[#333]'}`}>
        <div
          className={`flex-1 text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
            activeTab === 'chat'
              ? theme === 'light'
                ? 'text-gray-900 border-b-4 border-blue-500'
                : 'text-white border-b-4 border-[#297FFF]'
              : theme === 'light'
                ? 'text-gray-600 hover:text-gray-900'
                : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('chat')}>
          Chat
        </div>
        <div
          className={`flex-1 text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
            activeTab === 'quiz'
              ? theme === 'light'
                ? 'text-gray-900 border-b-4 border-blue-500'
                : 'text-white border-b-4 border-[#297FFF]'
              : theme === 'light'
                ? 'text-gray-600 hover:text-gray-900'
                : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('quiz')}>
          Quiz
        </div>
      </div>
      <div className="flex-1">
        {activeTab === 'chat' && <ChatTab articleData={articleData} apiBaseUrl={API_BASE_URL} theme={theme} />}
        {activeTab === 'quiz' && (
          <QuizTab
            articleData={articleData}
            apiBaseUrl={API_BASE_URL}
            theme={theme}
            quizMessages={quizMessages}
            setQuizMessages={setQuizMessages}
            currentQuizData={currentQuizData}
            setCurrentQuizData={setCurrentQuizData}
            questionStates={questionStates}
            setQuestionStates={setQuestionStates}
          />
        )}
      </div>
    </div>
  );
};

export default App;
