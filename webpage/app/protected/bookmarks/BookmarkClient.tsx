'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Database } from '@/database.types';
import { Search } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

type Bookmark = Database['public']['Tables']['bookmark']['Row'] & {
  tags?: string[];
};

interface BookmarkClientProps {
  bookmarks: Bookmark[];
}

export default function BookmarkClient({ bookmarks }: BookmarkClientProps) {
  // State for the currently selected article URL
  const [selectedUrl, setSelectedUrl] = useState<string | null>(bookmarks[0]?.url || null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState<'title' | 'tags'>('title');

  const handleSelectArticle = (url: string) => {
    setSelectedUrl(url);
  };

  const handleBack = () => {
    setSelectedUrl(null);
  };

  // Filter bookmarks based on search query and type
  const filteredBookmarks = bookmarks.filter(bookmark => {
    const query = searchQuery.toLowerCase();
    if (searchType === 'title') {
      return bookmark.title.toLowerCase().includes(query);
    } else {
      return bookmark.tags?.some(tag => tag.toLowerCase().includes(query)) || false;
    }
  });

  // Find the selected bookmark if `selectedUrl` is set
  const selectedBookmark = bookmarks.find(bookmark => bookmark.url === selectedUrl);

  return (
    <div className="h-screen flex flex-col">
      <header className="p-4 border-b">
        <h1 className="text-3xl font-bold">My Bookmarks</h1>
      </header>

      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        <div
          className={`
            md:w-1/3 
            border-r border-gray-200 
            ${selectedUrl ? 'hidden md:block' : 'block'}
            overflow-y-auto
          `}>
          {/* Search controls */}
          <div className="p-4 space-y-2">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
                <Input
                  type="text"
                  placeholder={searchType === 'title' ? 'Search by title...' : 'Search by tags...'}
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select value={searchType} onValueChange={(value: 'title' | 'tags') => setSearchType(value)}>
                <SelectTrigger className="w-[120px]">
                  <SelectValue placeholder="Search by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="title">Title</SelectItem>
                  <SelectItem value="tags">Tags</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <ul className="divide-y divide-gray-200">
            {filteredBookmarks.map(bookmark => (
              <li
                key={bookmark.url}
                onClick={() => handleSelectArticle(bookmark.url)}
                className="p-4 cursor-pointer hover:bg-gray-100">
                <h2 className="text-lg font-semibold">{bookmark.title || 'Untitled Article'}</h2>
                <p className="text-sm text-blue-500 truncate">{bookmark.url}</p>
                {bookmark.tags && bookmark.tags.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {bookmark.tags.map((tag, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>

        {/* Right column: Article details */}
        <div className={`md:w-2/3 ${selectedUrl ? 'block' : 'hidden md:block'} overflow-y-auto`}>
          {selectedBookmark && (
            <div className="p-4">
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
                  {selectedBookmark.tags && selectedBookmark.tags.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {selectedBookmark.tags.map((tag, index) => (
                        <Badge key={index} variant="secondary">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
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
