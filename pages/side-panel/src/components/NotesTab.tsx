import React, { useEffect, useState, useRef } from 'react';
import { Session } from '@supabase/supabase-js';
import {
  HighlightData,
  CreateHighlightRequest,
  UpdateHighlightRequest,
  HighlightEventRuntime,
} from '@extension/shared';
import { getSupabaseClient } from '@extension/shared/lib/utils/supabaseClient';
interface NotesTabProps {
  theme: 'light' | 'dark';
  session: Session | null;
}

const supabase = getSupabaseClient();

const NotesTab: React.FC<NotesTabProps> = ({ theme, session }) => {
  const [highlights, setHighlights] = useState<HighlightData[]>([]);
  const [currentUrl, setCurrentUrl] = useState<string>('');
  const [editingCommentId, setEditingCommentId] = useState<number | null>(null);
  const [commentText, setCommentText] = useState<string>('');
  const notesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when notes change
  const scrollToBottom = () => {
    if (notesContainerRef.current) {
      notesContainerRef.current.scrollTop = notesContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [highlights]);

  async function createHighlight(highlight: CreateHighlightRequest) {
    if (!session) {
      console.error('NoteTab: No session found in storage for creating highlight');
      return;
    }
    // TODO: check if user_id needs to be passed
    const { data, error } = await supabase
      .from('highlight')
      .insert({ ...highlight, user_id: session.user.id })
      .select();
    if (error) {
      console.error('Error creating highlight:', error);
      return;
    }
    setHighlights(prevHighlights => [...prevHighlights, data[0]]);
    console.log('Successfully created highlight:', data);
  }

  async function updateHighlight(oldHighlight: HighlightData, newHighlight: HighlightData) {
    if (!session) {
      console.error('NoteTab: No session found in storage for updating highlight comment');
      return;
    }
    // only update fields that changed
    const diff: UpdateHighlightRequest = {};
    // Only allow updating color and comment
    if (newHighlight.color !== oldHighlight.color) {
      diff.color = newHighlight.color;
    }
    if (newHighlight.comment !== oldHighlight.comment) {
      diff.comment = newHighlight.comment;
    }
    if (Object.keys(diff).length === 0) {
      console.log('NoteTab: No changes to highlight, skipping update');
      return;
    }
    const { data, error } = await supabase.from('highlight').update(diff).eq('id', oldHighlight.id).select();
    if (error) {
      console.error('Error updating highlight:', error);
      return;
    }
    setHighlights(prevHighlights =>
      prevHighlights.map(highlight => (highlight.id === oldHighlight.id ? { ...highlight, ...diff } : highlight)),
    );
    console.log('Successfully updated highlight:', data);
  }

  async function deleteHighlight(id: number) {
    if (!session) {
      console.error('NoteTab: No session found in storage for deleting highlight');
      return;
    }
    const { data, error } = await supabase.from('highlight').delete().eq('id', id).select();
    if (error) {
      console.error('Error deleting highlight:', error);
      return;
    }
    setHighlights(prevHighlights => prevHighlights.filter(highlight => highlight.id !== id));
    console.log('Successfully deleted highlight:', data);
  }

  const loadHighlights = async (url: string) => {
    if (!session) {
      console.error('NoteTab: No session found in storage for loading highlights');
      return;
    }
    const { data, error } = await supabase
      .from('highlight')
      .select()
      .eq('url', url)
      .eq('user_id', session.user.id)
      .select();
    if (error) {
      console.error('Error loading highlights:', error);
      return;
    }
    setHighlights(data);
    console.log('NoteTab: Retrieved stored highlights:', data);
  };

  const handleNewHighlight = React.useCallback(
    async (highlightData: CreateHighlightRequest) => {
      if (!session) {
        console.error('NoteTab: No session found for creating highlight');
        return;
      }
      await createHighlight(highlightData);
    },
    [session],
  );

  const handleDeleteHighlight = async (id: number, local_id: string) => {
    await deleteHighlight(id);
    // create message to delete highlight from content-runtime
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (tabs && tabs.length > 0) {
        const tabId = tabs[0].id;
        if (!tabId) {
          console.error('handleDeleteHighlight: No tabId found');
          return;
        }
        const message = {
          type: 'DELETE_HIGHLIGHT',
          local_id, // now at the top level
        };
        chrome.tabs.sendMessage(tabId, message, response => {
          console.log('Response from content script:', response);
        });
      }
    });
  };

  // Load highlights when component mounts
  useEffect(() => {
    const loadInitialHighlights = async () => {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs[0]?.url) {
        const url = tabs[0].url;
        setCurrentUrl(url);
        await loadHighlights(url);
      }
    };
    loadInitialHighlights();
  }, []); // Run only on mount

  // Listen for highlight events
  useEffect(() => {
    if (!session) return; // Don't set up listener without session

    const highlightListener = (
      message: any,
      sender: chrome.runtime.MessageSender,
      sendResponse: (response?: any) => void,
    ) => {
      console.log('NoteTab: Received message:', message);
      console.log(Object.keys(message));

      if (message.type === 'HIGHLIGHT_TEXT_RUNTIME') {
        const customMessage = message as HighlightEventRuntime;
        if (customMessage.url !== currentUrl) {
          console.log('NoteTab: Ignoring highlight data for different URL');
          return;
        }
        console.log('NoteTab: Processing highlight data:', message);
        const { type, ...highlightData } = message;
        handleNewHighlight(highlightData);
      }
      // Must return true if response is sent asynchronously
      return true;
    };

    console.log('NoteTab: Setting up highlight listener');
    chrome.runtime.onMessage.addListener(highlightListener);

    return () => {
      console.log('NoteTab: Removing highlight listener');
      chrome.runtime.onMessage.removeListener(highlightListener);
    };
  }, [session, handleNewHighlight]); // Only depend on session and memoized handler

  const saveComment = async (id: number) => {
    if (!session) {
      console.error('NoteTab: No session found in storage for saving highlight comment');
      return;
    }
    if (editingCommentId !== id) {
      console.error('NoteTab: Saving comment for wrong highlight');
      return;
    }
    const oldHighlight = highlights.find(highlight => highlight.id === id);
    if (!oldHighlight) {
      console.error('NoteTab: Saving comment for non-existent highlight');
      return;
    }
    const newHighlight = { ...oldHighlight, comment: commentText };
    await updateHighlight(oldHighlight, newHighlight);
    // update old highlight with new comment
    setHighlights(highlights.map(highlight => (highlight.id === id ? newHighlight : highlight)));
    setEditingCommentId(null);
    setCommentText('');
  };

  const scrollToHighlight = (highlight: HighlightData) => {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (tabs[0]?.id) {
        const tabId = tabs[0].id;
        if (!tabId) {
          console.error('scrollToHighlight: No tabId found');
          return;
        }
        const message = {
          type: 'SCROLL_TO_HIGHLIGHT',
          local_id: highlight.local_id,
        };
        chrome.tabs.sendMessage(tabId, message, response => {
          console.log('Response from content script:', response);
        });
      }
    });
  };

  const startEditing = (highlight: HighlightData) => {
    setEditingCommentId(highlight.id);
    setCommentText(highlight.comment || '');
  };

  return (
    <div
      className={`h-[calc(100vh-220px)] overflow-y-auto p-4 ${
        theme === 'light' ? 'bg-white text-gray-900' : 'bg-[#1E1E1E] text-gray-100'
      }`}>
      {highlights.length === 0 ? (
        <div className={`text-center py-8 ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
          No highlights found on this page
        </div>
      ) : (
        <div className="space-y-4" ref={notesContainerRef}>
          {highlights.map(highlight => (
            <div
              key={highlight.id}
              className={`p-4 rounded-lg transition-colors cursor-pointer ${
                theme === 'light' ? 'bg-gray-50 hover:bg-gray-100' : 'bg-[#2D2D2D] hover:bg-[#3D3D3D]'
              }`}
              onClick={() => scrollToHighlight(highlight)}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: highlight.color }} />
                  <span className={`text-xs ${theme === 'light' ? 'text-gray-500' : 'text-gray-400'}`}>
                    {new Date(highlight.created_at).toLocaleString()}
                  </span>
                </div>
                <button
                  onClick={e => {
                    e.stopPropagation();
                    handleDeleteHighlight(highlight.id, highlight.local_id);
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
                      className={`w-full p-2 rounded border resize-none ${
                        theme === 'light'
                          ? 'border-gray-300 bg-white text-gray-900 placeholder-gray-400'
                          : 'border-gray-600 bg-[#1E1E1E] text-gray-100 placeholder-gray-400'
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
                      theme === 'light'
                        ? 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                        : 'bg-[#3D3D3D] text-gray-200 hover:bg-[#4D4D4D]'
                    } cursor-text`}>
                    {highlight.comment || (
                      <span className={theme === 'light' ? 'text-gray-500' : 'text-gray-400'}>Add a comment...</span>
                    )}
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
