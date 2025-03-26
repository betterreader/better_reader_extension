// src/components/ChatTab.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ArticleData } from '../SidePanel';
import { enhancedChat } from '@extension/shared/lib/utils/vectorSearch';

interface ChatTabProps {
  articleData: ArticleData | null;
  apiBaseUrl: string;
  theme: 'light' | 'dark';
}

interface Message {
  sender: 'bot' | 'user';
  text: string;
  usedGeneralKnowledge?: boolean;
  sources?: Array<{ title: string; url: string }>;
  isTyping?: boolean;
}

interface Suggestion {
  text: string;
  onClick: () => void;
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
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [useEnhancedChat, setUseEnhancedChat] = useState<boolean>(true);

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
    const typingMessage: Message = { sender: 'bot', text: '...', isTyping: true };
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
    const typingMessage: Message = { sender: 'bot', text: '...', isTyping: true };
    setMessages(prev => [...prev, typingMessage]);

    if (useEnhancedChat) {
      // Use the enhanced chat that searches across all articles
      sendEnhancedChatMessage(text);
    } else if (!articleData || !articleData.content) {
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
    } else {
      // Use the regular chat that only works with the current article
      sendRegularChatMessage(text);
    }
  };

  const sendEnhancedChatMessage = async (text: string) => {
    try {
      // Get conversation history from previous messages
      const conversationHistory = messages
        .filter(msg => !msg.isTyping) // Skip typing indicators
        .map(msg => ({
          role: msg.sender === 'user' ? ('user' as const) : ('assistant' as const),
          content: msg.text,
        }));

      // Call the enhanced chat API
      const response = await enhancedChat(
        text,
        conversationId,
        conversationHistory,
        articleData?.url, // Use URL instead of ID since it's available and can work as an identifier
        articleData?.content,
      );

      // Update conversation ID for continuity
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }

      // Set suggestions if available
      if (response.suggestions && response.suggestions.length > 0) {
        setSuggestions(response.suggestions);
      } else {
        setSuggestions([]);
      }

      // Update messages with response
      setMessages(prev => {
        const newMessages = [...prev];
        // Remove typing indicator (last message)
        newMessages.pop();
        if (response.response) {
          newMessages.push({
            sender: 'bot',
            text: response.response,
            sources: response.sources,
          });
        } else if (response.error) {
          newMessages.push({ sender: 'bot', text: `Error: ${response.error}` });
        }
        return newMessages;
      });
    } catch (error) {
      console.error('Error in enhanced chat:', error);
      setMessages(prev => {
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

  const sendRegularChatMessage = (text: string) => {
    const requestData = {
      message: text,
      articleContent: articleData!.content,
      articleTitle: articleData!.title,
      articleUrl: articleData!.url,
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

  // Toggle between enhanced mode and regular mode
  const toggleChatMode = () => {
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

  // Function to handle clicking a suggestion
  const handleSuggestionClick = (suggestion: string) => {
    setInputValue(suggestion);
    // Optionally, auto-send the suggestion
    // sendMessageWithText(suggestion);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3 min-h-0">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg ${
              msg.sender === 'user' ? 'bg-blue-100 dark:bg-blue-900 ml-auto' : 'bg-gray-100 dark:bg-gray-800 mr-auto'
            } ${theme === 'dark' ? 'text-white' : 'text-gray-800'} max-w-[85%]`}>
            <div>{msg.text}</div>

            {/* Show general knowledge disclaimer if relevant */}
            {msg.usedGeneralKnowledge && (
              <div className="text-xs mt-2 italic text-gray-500 dark:text-gray-400">
                This response includes general knowledge not found in the article.
              </div>
            )}

            {/* Show sources if available */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="text-xs mt-2 text-gray-500 dark:text-gray-400">
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

        {/* Suggestions */}
        {suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                className="px-3 py-1 text-sm bg-gray-200 dark:bg-gray-700 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                onClick={() => handleSuggestionClick(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        {/* Mode toggle */}
        <div className="flex justify-between items-center mb-2">
          <button
            className={`text-xs px-2 py-1 rounded ${
              useEnhancedChat
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
            }`}
            onClick={toggleChatMode}>
            {useEnhancedChat ? 'Enhanced Mode (All Articles)' : 'Regular Mode (Current Article)'}
          </button>
        </div>
        <div className="flex items-center gap-2">
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
