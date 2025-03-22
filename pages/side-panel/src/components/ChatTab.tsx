// src/components/ChatTab.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ArticleData } from '../SidePanel';

interface ChatTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
}

interface Message {
  sender: 'bot' | 'user';
  text: string;
}

const ChatTab: React.FC<ChatTabProps> = ({ articleData, apiBaseUrl, theme }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'Hi there! I can help you understand this article better. Ask me anything about it.',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Listen for selected text from context menu
  useEffect(() => {
    // Check for stored selected text first (for when side panel is opened by context menu)
    chrome.storage.local.get(['selectedTextForExplanation'], result => {
      if (result.selectedTextForExplanation) {
        const { text, timestamp } = result.selectedTextForExplanation;

        // Only process if it's recent (within last 5 seconds)
        const now = Date.now();
        if (now - timestamp < 5000) {
          // Clear the stored text to prevent duplicate processing
          chrome.storage.local.remove(['selectedTextForExplanation']);

          // Process the selected text
          handleSelectedText(text);
        }
      }
    });

    // Listen for messages from background script
    const messageListener = (message: any) => {
      if (message.action === 'sendSelectedText' && message.text) {
        handleSelectedText(message.text);
      }
    };

    chrome.runtime.onMessage.addListener(messageListener);

    // Clean up listener on unmount
    return () => {
      chrome.runtime.onMessage.removeListener(messageListener);
    };
  }, [articleData]); // Re-run when articleData changes

  const handleSelectedText = (selectedText: string) => {
    if (!selectedText.trim()) return;

    const messageText = `Explain: ${selectedText}`;
    setInputValue(messageText);

    // Automatically send the message
    sendMessageWithText(messageText);
  };

  const sendMessage = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    sendMessageWithText(trimmed);
  };

  const sendMessageWithText = (text: string) => {
    // Add user message
    setMessages(prev => [...prev, { sender: 'user', text: text }]);
    setInputValue('');

    // Add typing indicator for bot
    const typingMessage: Message = { sender: 'bot', text: '...' };
    setMessages(prev => [...prev, typingMessage]);

    if (!articleData || !articleData.content) {
      // Replace typing indicator with an error message
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          sender: 'bot',
          text: "Sorry, I couldn't extract any content from this page.",
        };
        return newMessages;
      });
      return;
    }

    const requestData = {
      message: text,
      articleContent: articleData.content,
      articleTitle: articleData.title,
      articleUrl: articleData.url,
    };

    fetch(`${apiBaseUrl}/api/chat`, {
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
        setMessages(prev => {
          const newMessages = [...prev];
          // Remove typing indicator (last message)
          newMessages.pop();
          if (data.response) {
            newMessages.push({ sender: 'bot', text: data.response });
          } else if (data.error) {
            newMessages.push({ sender: 'bot', text: `Error: ${data.error}` });
          }
          return newMessages;
        });
      })
      .catch(error => {
        console.error('Error:', error);
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages.pop();
          newMessages.push({ sender: 'bot', text: 'Sorry, I encountered an error while processing your request.' });
          return newMessages;
        });
      });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg max-w-[80%] break-words leading-6 shadow-sm ${
              msg.sender === 'bot'
                ? theme === 'light'
                  ? 'self-start bg-gray-100 text-gray-800'
                  : 'self-start bg-[#252526] text-[#D4D4D4]'
                : 'self-end bg-[#0078D4] text-white'
            }`}>
            {msg.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div
        className={`flex p-3 border-t ${
          theme === 'light' ? 'bg-white border-gray-200' : 'bg-[#1E1E1E] border-[#333]'
        }`}>
        <textarea
          className={`flex-1 p-3 border rounded outline-none resize-none ${
            theme === 'light'
              ? 'bg-white border-gray-300 text-gray-900 focus:border-blue-500'
              : 'border-[#3C3C3C] bg-[#252526] text-[#D4D4D4] focus:border-[#0078D4]'
          }`}
          placeholder="Type your message..."
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className="bg-[#0078D4] text-white rounded w-10 h-10 ml-2 flex items-center justify-center hover:bg-[#005A9E] transition-colors"
          onClick={sendMessage}>
          {/* SVG icon */}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M22 2L11 13" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path
              d="M22 2L15 22L11 13L2 9L22 2Z"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default ChatTab;
