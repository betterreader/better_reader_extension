import React, { useState, useEffect } from 'react';
import ChatTab from '@src/components/ChatTab';
import QuizTab from '@src/components/QuizTab';
import NotesTab from '@src/components/NotesTab';
import ResearchTab from '@src/components/ResearchTab';
import SummaryTab from '@src/components/SummaryTab';
import { exampleThemeStorage } from '@extension/storage';
import SettingsDropdown from '@src/components/SettingsDropdown';
import { useStorage } from '@extension/shared';
import { Session } from '@supabase/supabase-js';

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

interface SidePanelProps {
  session: Session | null;
}

const API_BASE_URL = 'http://localhost:5007';

const App: React.FC<SidePanelProps> = ({ session }) => {
  const [activeTab, setActiveTab] = useState<'chat' | 'quiz' | 'notes' | 'research' | 'summary'>('chat');
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
    const resetAllStates = () => {
      // Reset quiz states
      setQuizMessages([
        {
          sender: 'bot',
          text: 'I can generate quiz questions about this article. Try clicking "Generate Quiz" below!',
        },
      ]);
      setCurrentQuizData(null);
      setQuestionStates([]);

      // Reset active tab to chat
      setActiveTab('chat');
    };

    const fetchArticleContent = () => {
      chrome.runtime.sendMessage({ action: 'getArticleContent' }, (response: any) => {
        if (response && response.content) {
          setArticleData({
            content: response.content,
            title: response.title,
            url: response.url,
          });
        } else {
          console.error('Failed to get article content:', response?.error);
          setArticleData(null);
        }
        // Reset all states when new content is fetched
        resetAllStates();
      });
    };

    // Fetch initial content
    fetchArticleContent();

    // Listen for tab updates to fetch new content
    const tabUpdateListener = (tabId: number, changeInfo: chrome.tabs.TabChangeInfo) => {
      if (changeInfo.status === 'complete') {
        fetchArticleContent();
      }
    };
    chrome.tabs.onUpdated.addListener(tabUpdateListener);

    // Listen for tab switches to fetch new content
    // Listen for tab switches to fetch new content
    const tabSwitchListener = (activeInfo: chrome.tabs.TabActiveInfo) => {
      console.log('SidePanel: Tab switch detected, fetching new content');
      fetchArticleContent();
    };

    chrome.tabs.onActivated.addListener(tabSwitchListener);

    // Listen for messages from background script to switch to chat tab
    // when text is selected for explanation
    const sendSelectedTextListener = (message: any) => {
      if (message.action === 'sendSelectedText' && message.text) {
        // Switch to chat tab when text is selected for explanation
        setActiveTab('chat');
      }
    };

    chrome.runtime.onMessage.addListener(sendSelectedTextListener);

    // Also check if there's stored selected text
    chrome.storage.local.get(['selectedTextForExplanation'], result => {
      if (result.selectedTextForExplanation) {
        // Switch to chat tab if there's selected text waiting
        setActiveTab('chat');
      }
    });

    // Clean up listeners on unmount
    return () => {
      chrome.runtime.onMessage.removeListener(sendSelectedTextListener);
      chrome.tabs.onUpdated.removeListener(tabUpdateListener);
      chrome.tabs.onActivated.removeListener(tabSwitchListener);
    };
  }, []);

  return (
    // Changed fixed width and height to fluid classes
    <div
      className={`w-full h-screen rounded-t-lg overflow-hidden flex flex-col ${
        theme === 'light' ? 'bg-white' : 'bg-[#1E1E1E]'
      }`}>
      <header
        className={`flex flex-col p-4 ${theme === 'light' ? 'bg-gray-100 text-gray-900' : 'bg-[#1A1A1A] text-white'}`}>
        <div className="flex justify-between items-center mb-2">
          <span className="text-2xl font-extrabold">
            <a href="http://localhost:3000" target="_blank" rel="noopener noreferrer">
              BetterReader
            </a>
          </span>
          <SettingsDropdown theme={theme} />
        </div>
        <div className={`text-base font-medium truncate ${theme === 'light' ? 'text-gray-600' : 'text-gray-300'}`}>
          {articleData?.title || 'Article Not Found'}
        </div>
      </header>
      <div
        className={`flex justify-between border-b px-4 ${theme === 'light' ? 'bg-gray-100 border-gray-200' : 'bg-[#1A1A1A] border-[#333]'}`}>
        <div
          className={` text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
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
          className={` text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
            activeTab === 'summary'
              ? theme === 'light'
                ? 'text-gray-900 border-b-4 border-blue-500'
                : 'text-white border-b-4 border-[#297FFF]'
              : theme === 'light'
                ? 'text-gray-600 hover:text-gray-900'
                : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('summary')}>
          Summary
        </div>
        <div
          className={` text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
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
        <div
          className={` text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
            activeTab === 'notes'
              ? theme === 'light'
                ? 'text-gray-900 border-b-4 border-blue-500'
                : 'text-white border-b-4 border-[#297FFF]'
              : theme === 'light'
                ? 'text-gray-600 hover:text-gray-900'
                : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('notes')}>
          Notes
        </div>
        <div
          className={` text-center py-4 text-base font-semibold cursor-pointer transition-colors ${
            activeTab === 'research'
              ? theme === 'light'
                ? 'text-gray-900 border-b-4 border-blue-500'
                : 'text-white border-b-4 border-[#297FFF]'
              : theme === 'light'
                ? 'text-gray-600 hover:text-gray-900'
                : 'text-gray-500 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('research')}>
          Research
        </div>
      </div>
      <div className="flex-1">
        <div className={`h-full ${activeTab !== 'chat' ? 'hidden' : ''}`}>
          <ChatTab
            key={articleData?.url || 'no-article'}
            articleData={articleData}
            apiBaseUrl={API_BASE_URL}
            theme={theme}
          />
        </div>
        <div className={`h-full ${activeTab !== 'quiz' ? 'hidden' : ''}`}>
          <QuizTab
            key={articleData?.url || 'no-article'}
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
        </div>
        <div className={`h-full ${activeTab !== 'notes' ? 'hidden' : ''}`}>
          <NotesTab key={articleData?.url || 'no-article'} theme={theme} session={session} />
        </div>
        <div className={`h-full ${activeTab !== 'research' ? 'hidden' : ''}`}>
          <ResearchTab
            key={articleData?.url || 'no-article'}
            articleData={articleData}
            apiBaseUrl={API_BASE_URL}
            theme={theme}
          />
        </div>
        <div className={`h-full ${activeTab !== 'summary' ? 'hidden' : ''}`}>
          <SummaryTab
            key={articleData?.url || 'no-article'}
            articleData={articleData}
            apiBaseUrl={API_BASE_URL}
            theme={theme}
          />
        </div>
      </div>
    </div>
  );
};

export default App;
