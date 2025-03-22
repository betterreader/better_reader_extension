import React, { useEffect, useState } from 'react';

interface HighlightData {
  id: string;
  color: string;
  text: string;
  timestamp: number;
  range: SerializedRange;
  comment?: string;
}

interface SerializedRange {
  startXPath: string;
  startOffset: number;
  endXPath: string;
  endOffset: number;
}

interface StorageData {
  [url: string]: {
    highlights: { [key: string]: HighlightData };
  };
}

interface NotesTabProps {
  theme: 'light' | 'dark';
}

const NotesTab: React.FC<NotesTabProps> = ({ theme }) => {
  const [highlights, setHighlights] = useState<HighlightData[]>([]);
  const [currentUrl, setCurrentUrl] = useState<string>('');
  const [editingCommentId, setEditingCommentId] = useState<string | null>(null);
  const [commentText, setCommentText] = useState<string>('');

  const loadHighlights = async (url: string) => {
    const result = (await chrome.storage.local.get(url)) as StorageData;
    const pageHighlights = result[url]?.highlights || {};
    const highlightArray = Object.values(pageHighlights).sort((a, b) => b.timestamp - a.timestamp);
    setHighlights(highlightArray);
  };

  useEffect(() => {
    // Get current tab URL and load initial highlights
    chrome.tabs.query({ active: true, currentWindow: true }, async tabs => {
      if (tabs[0]?.url) {
        setCurrentUrl(tabs[0].url);
        await loadHighlights(tabs[0].url);
      }
    });

    // Listen for storage changes
    const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }) => {
      // Only update if the change is for the current URL
      if (changes[currentUrl]) {
        loadHighlights(currentUrl);
      }
    };

    // Listen for new highlight events from the content script
    const handleNewHighlight = () => {
      if (currentUrl) {
        loadHighlights(currentUrl);
      }
    };

    chrome.storage.onChanged.addListener(handleStorageChange);
    window.addEventListener('HIGHLIGHT_TEXT', handleNewHighlight);

    return () => {
      chrome.storage.onChanged.removeListener(handleStorageChange);
      window.removeEventListener('HIGHLIGHT_TEXT', handleNewHighlight);
    };
  }, [currentUrl]);

  const deleteHighlight = async (id: string) => {
    const result = (await chrome.storage.local.get(currentUrl)) as StorageData;
    const pageData = result[currentUrl] || { highlights: {} };

    delete pageData.highlights[id];

    await chrome.storage.local.set({ [currentUrl]: pageData });

    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          type: 'DELETE_HIGHLIGHT',
          highlightId: id,
        });
      }
    });

    await loadHighlights(currentUrl);
  };

  const saveComment = async (id: string) => {
    const result = (await chrome.storage.local.get(currentUrl)) as StorageData;
    const pageData = result[currentUrl] || { highlights: {} };

    if (pageData.highlights[id]) {
      pageData.highlights[id].comment = commentText;
      await chrome.storage.local.set({ [currentUrl]: pageData });

      await loadHighlights(currentUrl);
      setEditingCommentId(null);
      setCommentText('');
    }
  };

  const scrollToHighlight = (highlight: HighlightData) => {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          type: 'SCROLL_TO_HIGHLIGHT',
          highlightId: highlight.id,
        });
      }
    });
  };

  const startEditing = (highlight: HighlightData) => {
    setEditingCommentId(highlight.id);
    setCommentText(highlight.comment || '');
  };

  return (
    <div className={`h-full overflow-y-auto p-4 ${theme === 'light' ? 'bg-white' : 'bg-[#1E1E1E]'}`}>
      {highlights.length === 0 ? (
        <div className={`text-center py-8 ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
          No highlights found on this page
        </div>
      ) : (
        <div className="space-y-4">
          {highlights.map(highlight => (
            <div
              key={highlight.id}
              className={`p-4 rounded-lg ${
                theme === 'light' ? 'bg-gray-50 hover:bg-gray-100' : 'bg-[#2D2D2D] hover:bg-[#3D3D3D]'
              } transition-colors cursor-pointer`}
              onClick={() => scrollToHighlight(highlight)}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: highlight.color }} />
                  <span className={`text-xs ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
                    {new Date(highlight.timestamp).toLocaleString()}
                  </span>
                </div>
                <button
                  onClick={e => {
                    e.stopPropagation();
                    deleteHighlight(highlight.id);
                  }}
                  className={`text-sm px-2 py-1 rounded ${
                    theme === 'light' ? 'text-red-600 hover:bg-red-50' : 'text-red-400 hover:bg-red-900'
                  }`}>
                  Delete
                </button>
              </div>
              <p className={theme === 'light' ? 'text-gray-900' : 'text-gray-100'}>{highlight.text}</p>
              <div className="mt-2">
                {editingCommentId === highlight.id ? (
                  <div className="mt-2" onClick={e => e.stopPropagation()}>
                    <textarea
                      value={commentText}
                      onChange={e => setCommentText(e.target.value)}
                      className={`w-full p-2 rounded border ${
                        theme === 'light' ? 'border-gray-300 bg-white' : 'border-gray-600 bg-[#1E1E1E]'
                      }`}
                      rows={3}
                      placeholder="Add a comment..."
                    />
                    <div className="flex justify-end gap-2 mt-2">
                      <button
                        onClick={() => {
                          setEditingCommentId(null);
                          setCommentText('');
                        }}
                        className={`px-3 py-1 rounded ${
                          theme === 'light' ? 'text-gray-600 hover:bg-gray-100' : 'text-gray-400 hover:bg-[#3D3D3D]'
                        }`}>
                        Cancel
                      </button>
                      <button
                        onClick={() => saveComment(highlight.id)}
                        className={`px-3 py-1 rounded ${
                          theme === 'light'
                            ? 'bg-blue-500 text-white hover:bg-blue-600'
                            : 'bg-[#297FFF] text-white hover:bg-[#1E6FEF]'
                        }`}>
                        Save
                      </button>
                    </div>
                  </div>
                ) : (
                  <div
                    onClick={e => {
                      e.stopPropagation();
                      startEditing(highlight);
                    }}
                    className={`mt-2 p-2 rounded ${
                      theme === 'light' ? 'bg-gray-100 hover:bg-gray-200' : 'bg-[#3D3D3D] hover:bg-[#4D4D4D]'
                    } cursor-text`}>
                    {highlight.comment || 'Add a comment...'}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default NotesTab;
