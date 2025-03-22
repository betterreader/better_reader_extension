import React, { useEffect, useState } from 'react';

interface Highlight {
  id: string;
  color: string;
  text: string;
  timestamp: number;
}

export const Highlighter: React.FC = () => {
  console.log('Rendering content-ui');
  const [highlights, setHighlights] = useState<Highlight[]>([]);

  useEffect(() => {
    const handleHighlight = (event: CustomEvent) => {
      const { color, range } = event.detail;

      // Create new highlight entry
      const newHighlight: Highlight = {
        id: Math.random().toString(36).substr(2, 9),
        color,
        text: window.getSelection()?.toString() || '',
        timestamp: Date.now(),
      };

      setHighlights(prev => [...prev, newHighlight]);
    };

    // Listen for highlight events from content script
    window.addEventListener('HIGHLIGHT_TEXT', handleHighlight as EventListener);

    return () => {
      window.removeEventListener('HIGHLIGHT_TEXT', handleHighlight as EventListener);
    };
  }, []);

  const removeHighlight = (id: string) => {
    setHighlights(prev => prev.filter(h => h.id !== id));
    // TODO: Remove highlight from DOM
  };

  if (highlights.length === 0) return null;

  return (
    // <div className="fixed bottom-4 right-4 bg-white p-4 rounded-lg shadow-lg max-w-md">
    //   <h3 className="text-lg font-semibold mb-2">Highlights</h3>
    //   <div className="max-h-60 overflow-y-auto">
    //     {highlights.map(highlight => (
    //       <div key={highlight.id} className="flex items-center gap-2 p-2 hover:bg-gray-50 rounded">
    //         <div className="w-4 h-4 rounded-full" style={{ backgroundColor: highlight.color }} />
    //         <span className="flex-1 truncate">{highlight.text}</span>
    //         <button onClick={() => removeHighlight(highlight.id)} className="text-gray-400 hover:text-gray-600">
    //           ×
    //         </button>
    //       </div>
    //     ))}
    //   </div>
    // </div>
    <div></div>
  );
};
