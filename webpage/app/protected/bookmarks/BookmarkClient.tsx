'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Database } from '@/database.types';
import { Search } from 'lucide-react';

type Bookmark = Database['public']['Tables']['bookmark']['Row'];

interface BookmarkClientProps {
  bookmarks: Bookmark[];
}

export default function BookmarkClient({ bookmarks }: BookmarkClientProps) {
  // State for the currently selected article URL
  const [selectedUrl, setSelectedUrl] = useState<string | null>(bookmarks[0]?.url || null);
  const [searchQuery, setSearchQuery] = useState('');

  const handleSelectArticle = (url: string) => {
    setSelectedUrl(url);
  };

  const handleBack = () => {
    setSelectedUrl(null);
  };

  // Filter highlights based on search query
  const filteredHighlights = bookmarks.filter(bookmark =>
    bookmark.title.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  // Find the selected group if `selectedUrl` is set
  const selectedBookmark = bookmarks.find(bookmark => bookmark.url === selectedUrl);

  return (
    // 1) Fix the page to full viewport height
    <div className="h-screen flex flex-col">
      {/* Top header area */}
      <header className="p-4 border-b">
        <h1 className="text-3xl font-bold">My Bookmarks</h1>
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
            {bookmarks.map(bookmark => (
              <li
                key={bookmark.url}
                onClick={() => handleSelectArticle(bookmark.url)}
                className="p-4 cursor-pointer hover:bg-gray-100">
                <h2 className="text-lg font-semibold">{bookmark.title || 'Untitled Article'}</h2>
                <p className="text-sm text-blue-500 truncate">{bookmark.url}</p>
              </li>
            ))}
          </ul>
        </div>

        {/* Right column: Highlights of the selected article */}
        <div className={`md:w-2/3 ${selectedUrl ? 'block' : 'hidden md:block'} overflow-y-auto`}>
          {selectedBookmark && (
            <div className="p-4">
              {/* “Back” button on mobile to return to article list */}
              <button
                onClick={handleBack}
                className="mb-4 inline-block md:hidden bg-blue-500 text-white px-4 py-2 rounded">
                Back
              </button>

              <Card className="p-6">
                <div className="mb-4">
                  <h2 className="text-xl font-semibold mb-2">{selectedBookmark.title || 'Untitled Article'}</h2>
                  <a
                    href={selectedBookmark.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-500 hover:underline">
                    {selectedBookmark.url}
                  </a>
                </div>
                <div className="space-y-4">
                  <div className="text-base mb-2">
                    {selectedBookmark.content.slice(0, Math.min(selectedBookmark.content.length, 500)) + '...'}
                  </div>
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
