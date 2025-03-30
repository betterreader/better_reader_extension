// src/components/ChatTab.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ArticleData } from '../SidePanel';
import { enhancedChat } from '@extension/shared/lib/utils/vectorSearch';

interface ChatTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
}

export interface Message {
  sender: 'bot' | 'user';
  text: string;
  usedGeneralKnowledge?: boolean;
  sources?: Array<{ title: string; url: string }>;
  isTyping?: boolean;
  isTeacherMode?: boolean;
}

const ChatTab: React.FC<ChatTabProps> = ({ articleData, apiBaseUrl, theme }) => {
  // Normal conversation messages (kept unchanged for non-teacher mode)
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'bot',
      text: 'Hi there! I can help you understand this article better. Ask me anything about it.',
    },
  ]);
  // Teacher mode conversation messages
  const [teacherMessages, setTeacherMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const lastProcessedTextRef = useRef<string>('');
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [useEnhancedChat, setUseEnhancedChat] = useState<boolean>(false);
  const [useTeacherMode, setUseTeacherMode] = useState<boolean>(false);
  const [goDeeper, setGoDeeper] = useState<boolean>(false);

  // Always scroll to the bottom on new messages.
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, teacherMessages]);

  // Display conversation based on mode.
  const displayedMessages = useTeacherMode ? teacherMessages : messages;

  // --- Helper: Update state based on current mode ---
  const updateMessageState = (updateFn: (prev: Message[]) => Message[], isTeacher: boolean) => {
    if (isTeacher) {
      setTeacherMessages(updateFn);
    } else {
      setMessages(updateFn);
    }
  };

  // --- Toggling Teacher Mode ---
  const toggleTeacherMode = () => {
    if (!useTeacherMode && teacherMessages.length === 0) {
      // Only add the activation message once (as the first teacher message)
      setTeacherMessages([
        {
          sender: 'bot',
          text: 'Teacher mode activated. I will help you understand concepts through guided questioning and discussion.',
          isTeacherMode: true,
        },
      ]);
    }
    // Toggle the flag.
    setUseTeacherMode(!useTeacherMode);
  };

  // --- Handling selected text (remains unchanged, except for mode-specific updates) ---
  useEffect(() => {
    const processSelectedText = (text: string, paragraph = '', title = '', mode = 'simple') => {
      if (text && text.trim()) {
        handleSelectedText(text, paragraph, title, mode);
      }
    };

    const checkStoredText = () => {
      chrome.storage.local.get(['selectedTextForExplanation'], result => {
        if (result.selectedTextForExplanation) {
          const { text, paragraph, title, mode, timestamp } = result.selectedTextForExplanation;
          const now = Date.now();
          if (now - timestamp < 10000) {
            chrome.storage.local.remove(['selectedTextForExplanation']);
            processSelectedText(text, paragraph, title, mode);
          }
        }
      });
    };

    checkStoredText();

    const messageListener = (
      message: any,
      sender: chrome.runtime.MessageSender,
      sendResponse: (response?: any) => void,
    ) => {
      console.log('Received message in ChatTab:', message);
      if (message.action === 'sendSelectedText' && message.text) {
        processSelectedText(message.text, message.paragraph, message.title, message.mode);
        sendResponse({ status: 'received' });
        return true;
      }
      return false;
    };

    chrome.runtime.onMessage.addListener(messageListener);
    const intervalId = setInterval(checkStoredText, 1000);

    return () => {
      chrome.runtime.onMessage.removeListener(messageListener);
      clearInterval(intervalId);
    };
  }, [articleData]);

  const handleSelectedText = (selectedText: string, paragraph = '', title = '', mode = 'simple') => {
    const textSignature = `${selectedText}:${mode}`;
    if (textSignature === lastProcessedTextRef.current) return;
    lastProcessedTextRef.current = textSignature;
    const formattedMessage = `Explain (${mode}): ${selectedText}`;
    const explanationContext = {
      text: selectedText,
      paragraph,
      title: title || articleData?.title || '',
      mode,
    };
    sendExplanationRequest(formattedMessage, explanationContext);
  };

  // --- Sending Explanation Requests ---
  const sendExplanationRequest = (formattedMessage: string, context: any) => {
    if (useTeacherMode) {
      setTeacherMessages(prev => [
        ...prev,
        { sender: 'user', text: formattedMessage },
        { sender: 'bot', text: '...', isTyping: true },
      ]);
    } else {
      setMessages(prev => [
        ...prev,
        { sender: 'user', text: formattedMessage },
        { sender: 'bot', text: '...', isTyping: true },
      ]);
    }

    if (!articleData || !articleData.content) {
      updateMessageState(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          sender: 'bot',
          text: "Sorry, I couldn't extract any content from this page.",
        };
        return newMessages;
      }, useTeacherMode);
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData),
    })
      .then(response => {
        if (!response.ok) throw new Error(`Network response was not ok: ${response.status}`);
        return response.json();
      })
      .then(data => {
        if (useTeacherMode) {
          setTeacherMessages(prev => {
            const newMessages = [...prev];
            newMessages.pop(); // remove typing indicator
            if (data.response) {
              newMessages.push({ sender: 'bot', text: data.response, isTeacherMode: true });
            } else if (data.error) {
              newMessages.push({ sender: 'bot', text: `Error: ${data.error}`, isTeacherMode: true });
            }
            return newMessages;
          });
        } else {
          setMessages(prev => {
            const newMessages = [...prev];
            newMessages.pop();
            if (data.response) {
              newMessages.push({ sender: 'bot', text: data.response });
            } else if (data.error) {
              newMessages.push({ sender: 'bot', text: `Error: ${data.error}` });
            }
            return newMessages;
          });
        }
      })
      .catch(error => {
        console.error('Error:', error);
        if (useTeacherMode) {
          setTeacherMessages(prev => {
            const newMessages = [...prev];
            newMessages.pop();
            newMessages.push({
              sender: 'bot',
              text: 'Sorry, I encountered an error while processing your request.',
              isTeacherMode: true,
            });
            return newMessages;
          });
        } else {
          setMessages(prev => {
            const newMessages = [...prev];
            newMessages.pop();
            newMessages.push({
              sender: 'bot',
              text: 'Sorry, I encountered an error while processing your request.',
            });
            return newMessages;
          });
        }
      });
  };

  // --- Sending regular user messages ---
  const sendMessageWithText = (text: string) => {
    setInputValue('');
    if (useTeacherMode) {
      setTeacherMessages(prev => [...prev, { sender: 'user', text }, { sender: 'bot', text: '...', isTyping: true }]);
      sendTeacherMessage(text);
    } else {
      setMessages(prev => [...prev, { sender: 'user', text }, { sender: 'bot', text: '...', isTyping: true }]);
      if (useEnhancedChat) {
        sendEnhancedChatMessage(text);
      } else if (!articleData || !articleData.content) {
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            sender: 'bot',
            text: "Sorry, I couldn't extract any content from this page.",
          };
          return newMessages;
        });
        return;
      } else {
        sendRegularChatMessage(text);
      }
    }
  };

  // --- Teacher Chat ---
  const sendTeacherMessage = async (text: string) => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/teacher_chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          articleContent: articleData!.content,
          articleTitle: articleData!.title,
          articleUrl: articleData!.url,
          conversationHistory: teacherMessages
            .filter(msg => !msg.isTyping)
            .map(msg => ({
              role: msg.sender === 'user' ? ('user' as const) : ('assistant' as const),
              content: msg.text,
            })),
        }),
      });
      if (!response.ok) throw new Error(`Network response was not ok: ${response.status}`);
      const data = await response.json();
      setTeacherMessages(prev => {
        const newMessages = [...prev];
        newMessages.pop(); // remove typing indicator
        if (data.response) {
          newMessages.push({ sender: 'bot', text: data.response, isTeacherMode: true });
        } else if (data.error) {
          newMessages.push({ sender: 'bot', text: `Error: ${data.error}`, isTeacherMode: true });
        }
        return newMessages;
      });
    } catch (error) {
      console.error('Error:', error);
      setTeacherMessages(prev => {
        const newMessages = [...prev];
        newMessages.pop();
        newMessages.push({
          sender: 'bot',
          text: 'Sorry, I encountered an error while processing your request.',
          isTeacherMode: true,
        });
        return newMessages;
      });
    }
  };

  // --- Regular Chat (non-teacher) ---
  const sendRegularChatMessage = (text: string) => {
    const requestData = {
      message: text,
      articleContent: articleData!.content,
      articleTitle: articleData!.title,
      articleUrl: articleData!.url,
      teacherMode: false,
    };

    fetch(`${apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData),
    })
      .then(response => {
        if (!response.ok) throw new Error(`Network response was not ok: ${response.status}`);
        return response.json();
      })
      .then(data => {
        setMessages(prev => {
          const newMessages = [...prev];
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

  // --- Enhanced Chat (non-teacher) ---
  const sendEnhancedChatMessage = async (text: string) => {
    try {
      // Check if article data is available
      if (!articleData || !articleData.content) {
        console.error('Enhanced chat called with no article data:', articleData);
        setMessages((prev: Message[]) => {
          const newMessages = [...prev];
          newMessages.pop();
          newMessages.push({
            sender: 'bot',
            text: "Sorry, I couldn't extract any content from this page. Please try refreshing the page or switching to regular chat mode.",
          });
          return newMessages;
        });
        return;
      }

      // Log article data before making the API call
      console.log('Enhanced chat sending article data:', {
        contentLength: articleData.content.length,
        contentSample: articleData.content.substring(0, 100) + '...',
        title: articleData.title,
        url: articleData.url,
        goDeeper: goDeeper,
      });

      const conversationHistory = messages
        .filter(msg => !msg.isTyping)
        .map(msg => ({
          role: msg.sender === 'user' ? ('user' as const) : ('assistant' as const),
          content: msg.text,
        }));

      console.log('Enhanced chat with article data:', {
        url: articleData.url,
        contentLength: articleData.content ? articleData.content.length : 0,
        goDeeper: goDeeper,
      });

      const response = await enhancedChat(
        text,
        conversationId,
        conversationHistory,
        articleData.url,
        articleData.content,
        articleData.title,
        goDeeper,
      );
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }
      setSuggestions(response.suggestions && response.suggestions.length > 0 ? response.suggestions : []);
      setMessages((prev: Message[]) => {
        const newMessages = [...prev];
        newMessages.pop();
        if (response.response) {
          newMessages.push({ sender: 'bot', text: response.response, sources: response.sources });
        } else if (response.error) {
          newMessages.push({ sender: 'bot', text: `Error: ${response.error}` });
        }
        return newMessages;
      });
    } catch (error) {
      console.error('Error in enhanced chat:', error);
      setMessages((prev: Message[]) => {
        const newMessages = [...prev];
        newMessages.pop();
        newMessages.push({
          sender: 'bot',
          text: 'Sorry, I encountered an error while searching across your articles.',
        });
        return newMessages;
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessage = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    if (trimmed.startsWith('Explain (') && trimmed.includes('): ')) {
      const modeMatch = trimmed.match(/Explain \(([^)]+)\):/);
      const mode = modeMatch ? modeMatch[1] : 'simple';
      const text = trimmed.split('): ')[1];
      const explanationContext = {
        text,
        paragraph: '',
        title: articleData?.title || '',
        mode,
      };
      sendExplanationRequest(trimmed, explanationContext);
    } else {
      sendMessageWithText(trimmed);
    }
  };

  // --- Toggle between enhanced and regular (non-teacher) modes ---
  const toggleChatMode = () => {
    // If trying to enable enhanced mode, check if article data is available
    if (!useEnhancedChat && (!articleData || !articleData.content)) {
      setMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          text: "Sorry, enhanced chat requires article content. Please make sure you're viewing an article or try refreshing the page.",
        },
      ]);
      return;
    }

    setUseEnhancedChat(!useEnhancedChat);
    setMessages(prev => [
      ...prev,
      {
        sender: 'bot',
        text: !useEnhancedChat
          ? 'Enhanced mode activated. I can now answer questions using your entire reading history across all articles.'
          : 'Regular mode activated. I will only answer questions about the current article.',
      },
    ]);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 min-h-0">
        {displayedMessages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg ${
              // For normal mode, use blue tones; for teacher mode, use green.
              useTeacherMode
                ? theme === 'dark'
                  ? 'bg-green-900'
                  : 'bg-green-100'
                : theme === 'dark'
                  ? 'bg-blue-900'
                  : 'bg-blue-100'
            } ${msg.sender === 'user' ? 'ml-auto' : 'mr-auto'} max-w-[85%] ${
              theme === 'dark' ? 'text-white' : 'text-gray-800'
            }`}>
            <div>{msg.text}</div>
            {msg.usedGeneralKnowledge && (
              <div className="text-xs mt-2 italic text-gray-500">
                This response includes general knowledge not found in the article.
              </div>
            )}
            {msg.sources && msg.sources.length > 0 && (
              <div className="text-xs mt-2 text-gray-500">
                <div className="font-medium">Sources:</div>
                <ul className="list-disc pl-4">
                  {msg.sources.map((source, idx) => (
                    <li key={idx}>
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-500 hover:underline">
                        {source.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                className={`px-3 py-1 text-sm ${
                  theme === 'dark'
                    ? 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                    : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                } rounded-full transition-colors`}
                onClick={() => setInputValue(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className={`p-4 border-t ${theme === 'dark' ? 'border-gray-700' : 'border-gray-200'}`}>
        <div className="flex justify-between items-center mb-2 gap-2">
          <div className="flex gap-2">
            <button
              className={`text-xs px-2 py-1 rounded ${
                useEnhancedChat
                  ? 'bg-blue-500 text-white'
                  : theme === 'dark'
                    ? 'bg-gray-700 text-gray-200'
                    : 'bg-gray-200 text-gray-800'
              }`}
              onClick={toggleChatMode}>
              {useEnhancedChat ? 'Enhanced Mode (All Articles)' : 'Regular Mode (Current Article)'}
            </button>
            {useEnhancedChat && (
              <button
                onClick={() => setGoDeeper(!goDeeper)}
                className={`px-2 py-1 text-xs rounded-md ${
                  goDeeper
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                }`}
                title="When enabled, includes all articles in the context instead of just the most relevant ones">
                {goDeeper ? 'Neural Retreival: ON' : 'Neural Retreival: OFF'}
              </button>
            )}
            <button
              className={`text-xs px-2 py-1 rounded ${
                // Teacher mode toggle button (always blue when off, green when on)
                useTeacherMode
                  ? theme === 'dark'
                    ? 'bg-green-700 text-white'
                    : 'bg-green-500 text-white'
                  : theme === 'dark'
                    ? 'bg-blue-700 text-white'
                    : 'bg-blue-500 text-white'
              }`}
              onClick={toggleTeacherMode}>
              {useTeacherMode ? '🧑‍🏫 Teacher Mode' : 'Regular Mode'}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Input box remains neutral (theme-dependent only) */}
          <textarea
            className={`w-full p-2 border rounded-lg resize-none ${
              theme === 'dark' ? 'bg-gray-800 text-white border-gray-700' : 'bg-white text-gray-800 border-gray-300'
            }`}
            placeholder="Ask a question about this article..."
            rows={2}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            onClick={sendMessage}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatTab;
