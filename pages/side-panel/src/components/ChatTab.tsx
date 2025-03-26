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
  usedGeneralKnowledge?: boolean;
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
  const lastProcessedTextRef = useRef<string>('');

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Listen for selected text from context menu
  useEffect(() => {
    // Function to handle selected text from storage or messages
    const processSelectedText = (text: string, paragraph: string = '', title: string = '', mode: string = 'simple') => {
      if (text && text.trim()) {
        handleSelectedText(text, paragraph, title, mode);
      }
    };

    // Check for stored selected text first (for when side panel is opened by context menu)
    const checkStoredText = () => {
      chrome.storage.local.get(['selectedTextForExplanation'], result => {
        if (result.selectedTextForExplanation) {
          const { text, paragraph, title, mode, timestamp } = result.selectedTextForExplanation;

          // Only process if it's recent (within last 10 seconds)
          const now = Date.now();
          if (now - timestamp < 10000) {
            // Clear the stored text to prevent duplicate processing
            chrome.storage.local.remove(['selectedTextForExplanation']);

            // Process the selected text with context and mode
            processSelectedText(text, paragraph, title, mode);
          }
        }
      });
    };

    // Check immediately when component mounts
    checkStoredText();

    // Listen for messages from background script
    const messageListener = (
      message: any,
      sender: chrome.runtime.MessageSender,
      sendResponse: (response?: any) => void,
    ) => {
      console.log('Received message in ChatTab:', message);

      if (message.action === 'sendSelectedText' && message.text) {
        processSelectedText(message.text, message.paragraph, message.title, message.mode);
        // Send a response to acknowledge receipt
        sendResponse({ status: 'received' });
        return true; // Indicates async response
      }

      // Return false for messages we don't handle
      return false;
    };

    // Add the listener
    chrome.runtime.onMessage.addListener(messageListener);

    // Set up a periodic check for new selected text (as a backup)
    const intervalId = setInterval(checkStoredText, 1000);

    // Clean up listener and interval on unmount
    return () => {
      chrome.runtime.onMessage.removeListener(messageListener);
      clearInterval(intervalId);
    };
  }, [articleData]); // Re-run when articleData changes

  const handleSelectedText = (
    selectedText: string,
    paragraph: string = '',
    title: string = '',
    mode: string = 'simple',
  ) => {
    // Prevent processing the same text multiple times
    const textSignature = `${selectedText}:${mode}`;
    if (textSignature === lastProcessedTextRef.current) {
      return;
    }
    lastProcessedTextRef.current = textSignature;

    // Format the message to indicate it's an explanation request with the selected mode
    const formattedMessage = `Explain (${mode}): ${selectedText}`;

    // Create explanation context
    const explanationContext = {
      text: selectedText,
      paragraph: paragraph,
      title: title || articleData?.title || '',
      mode: mode,
    };

    // Send the explanation request
    sendExplanationRequest(formattedMessage, explanationContext);
  };

  const sendExplanationRequest = (formattedMessage: string, context: any) => {
    // Add user message
    setMessages(prev => [...prev, { sender: 'user', text: formattedMessage }]);

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
      explanationRequest: true,
      text: context.text,
      paragraph: context.paragraph,
      title: context.title || articleData.title,
      mode: context.mode,
      articleContent: articleData.content,
      articleUrl: articleData.url,
    };

    fetch(`${apiBaseUrl}/api/explain`, {
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

  const sendMessage = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    // Check if this is an explanation request
    if (trimmed.startsWith('Explain (') && trimmed.includes('): ')) {
      // Extract the explanation context from the message
      const modeMatch = trimmed.match(/Explain \(([^)]+)\):/);
      const mode = modeMatch ? modeMatch[1] : 'simple';
      const text = trimmed.split('): ')[1];

      // Create explanation context
      const explanationContext = {
        text: text,
        paragraph: '',
        title: articleData?.title || '',
        mode: mode,
      };

      // Send as an explanation request
      sendExplanationRequest(trimmed, explanationContext);
    } else {
      // Send as a regular chat message
      sendMessageWithText(trimmed);
    }
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
            newMessages.push({ sender: 'bot', text: data.response, usedGeneralKnowledge: data.usedGeneralKnowledge });
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
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 max-h-[calc(100vh-220px)]">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg ${
              msg.sender === 'user'
                ? theme === 'light'
                  ? 'bg-blue-100 text-blue-800 ml-auto'
                  : 'bg-blue-900 text-blue-100 ml-auto'
                : theme === 'light'
                  ? 'bg-gray-100 text-gray-800'
                  : 'bg-[#252526] text-[#D4D4D4]'
            } ${msg.sender === 'user' ? 'max-w-[80%]' : 'max-w-[90%]'} break-words leading-6 shadow-sm`}>
            {msg.text}
            {msg.usedGeneralKnowledge && (
              <div
                className={`mt-2 pt-2 text-xs ${
                  theme === 'light'
                    ? 'text-gray-500 border-t border-gray-200'
                    : 'text-gray-400 border-t border-gray-700'
                }`}>
                <span
                  className={`inline-flex items-center px-2 py-1 rounded-full ${
                    theme === 'light' ? 'bg-purple-100 text-purple-800' : 'bg-purple-900 text-purple-100'
                  }`}>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-3 w-3 mr-1"
                    viewBox="0 0 20 20"
                    fill="currentColor">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z"
                      clipRule="evenodd"
                    />
                  </svg>
                  Answered using general knowledge
                </span>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className={`p-3 border-t ${theme === 'light' ? 'border-gray-200 bg-white' : 'border-[#333] bg-[#1E1E1E]'}`}>
        <div className="flex items-end gap-2">
          <textarea
            className={`flex-1 p-2 border rounded resize-none outline-none min-h-[40px] max-h-[120px] ${
              theme === 'light'
                ? 'border-gray-300 bg-white text-gray-900 focus:border-blue-500'
                : 'border-[#3C3C3C] bg-[#252526] text-[#D4D4D4] focus:border-[#0078D4]'
            }`}
            placeholder="Ask me about this article..."
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            style={{ height: 'auto', maxHeight: '120px' }}
            onInput={(e: React.FormEvent<HTMLTextAreaElement>) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = 'auto';
              target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
            }}
          />
          <button
            className="bg-[#0078D4] text-white p-2 rounded hover:bg-[#005A9E] transition-colors"
            onClick={sendMessage}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatTab;
