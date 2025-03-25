'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Database } from '@/database.types';
import { Search } from 'lucide-react';

type Highlight = Database['public']['Tables']['highlight']['Row'];

interface HighlightGroup {
  url: string;
  title: string;
  highlights: Highlight[];
}

interface HighlightsClientProps {
  groupedHighlights: HighlightGroup[];
}

export default function HighlightsClient({ groupedHighlights }: HighlightsClientProps) {
  // State for the currently selected article URL
  const [selectedUrl, setSelectedUrl] = useState<string | null>(groupedHighlights[0]?.url || null);
  const [searchQuery, setSearchQuery] = useState('');

  const handleSelectArticle = (url: string) => {
    setSelectedUrl(url);
  };

  const handleBack = () => {
    setSelectedUrl(null);
  };

  // Filter highlights based on search query
  const filteredHighlights = groupedHighlights.filter(group =>
    group.title.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  // Find the selected group if `selectedUrl` is set
  const selectedGroup = groupedHighlights.find(group => group.url === selectedUrl);

  return (
    // 1) Fix the page to full viewport height
    <div className="h-screen flex flex-col">
      {/* Top header area */}
      <header className="p-4 border-b">
        <h1 className="text-3xl font-bold">My Highlights</h1>
      </header>

      {/* 2) Content area that allows the columns to scroll independently */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left column: List of articles */}
        <div
          className={`
            md:w-1/3 
            border-r border-gray-200 
            ${selectedUrl ? 'hidden md:block' : 'block'}
            overflow-y-auto
          `}>
          {/* Search bar */}
          <div className="relative p-4">
            <Search className="absolute left-7 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
            <Input
              type="text"
              placeholder="Search articles..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          <ul className="divide-y divide-gray-200">
            {filteredHighlights.map(group => (
              <li
                key={group.url}
                onClick={() => handleSelectArticle(group.url)}
                className="p-4 cursor-pointer hover:bg-gray-100">
                <h2 className="text-lg font-semibold">{group.title || 'Untitled Article'}</h2>
                <p className="text-sm text-blue-500 truncate">{group.url}</p>
                <p className="text-sm text-gray-600 truncate">{group.highlights.length} highlights</p>
              </li>
            ))}
          </ul>
        </div>

        {/* Right column: Highlights of the selected article */}
        <div className={`md:w-2/3 ${selectedUrl ? 'block' : 'hidden md:block'} overflow-y-auto`}>
          {selectedGroup && (
            <div className="p-4">
              {/* “Back” button on mobile to return to article list */}
              <button
                onClick={handleBack}
                className="mb-4 inline-block md:hidden bg-blue-500 text-white px-4 py-2 rounded">
                Back
              </button>

              <Card className="p-6">
                <div className="mb-4">
                  <h2 className="text-xl font-semibold mb-2">{selectedGroup.title || 'Untitled Article'}</h2>
                  <a
                    href={selectedGroup.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-500 hover:underline">
                    {selectedGroup.url}
                  </a>
                  <p className="text-sm mt-2">{selectedGroup.highlights.length} highlights</p>
                </div>

                <div className="space-y-4">
                  {selectedGroup.highlights.map(highlight => (
                    <div
                      key={highlight.id}
                      className="border-l-4 pl-4 py-2"
                      style={{ borderColor: highlight.color || '#FFD700' }}>
                      <div className="text-base mb-2">{highlight.text}</div>
                      {highlight.comment && <div className="text-sm text-gray-600 mt-2">{highlight.comment}</div>}
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="secondary">{new Date(highlight.created_at).toLocaleDateString()}</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
