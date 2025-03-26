import React, { useState, useEffect } from 'react';
import { getSupabaseClient } from '@extension/shared/lib/utils/supabaseClient';

export interface ArticleData {
  content: string;
  title: string;
  url: string;
}

interface Props {
  theme: string;
  articleData: ArticleData | null;
}

const BookmarkToggle: React.FC<Props> = ({ theme, articleData }) => {
  if (!articleData) {
    return <div className="ml-1 text-sm">Cannot Save Article</div>;
  }

  const supabase = getSupabaseClient();
  const [bookmarked, setBookmarked] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchBookmark = async () => {
    setLoading(true);
    const user_id = (await supabase.auth.getUser()).data.user?.id;
    const { data, error } = await supabase
      .from('bookmark')
      .select('*')
      .eq('url', articleData.url)
      .eq('user_id', user_id);
    if (error) {
      console.error(error);
      setBookmarked(false);
      setLoading(false);
    } else {
      setBookmarked(data.length > 0);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBookmark();
  }, [articleData]);

  const addBookmark = async () => {
    const user_id = (await supabase.auth.getUser()).data.user?.id;
    console.log('ArticleData:', articleData);
    setLoading(true);
    const { data, error } = await supabase.from('bookmark').insert([
      {
        url: articleData.url,
        title: articleData.title,
        content: articleData.content,
        user_id: user_id,
      },
    ]);
    if (error) {
      console.error("Couldn't add bookmark:", error);
      setLoading(false);
    } else {
      setBookmarked(true);
      setLoading(false);
    }
  };

  const removeBookmark = async () => {
    setLoading(true);
    const user_id = (await supabase.auth.getUser()).data.user?.id;
    const { data, error } = await supabase
      .from('bookmark')
      .delete()
      .eq('url', articleData.url)
      .eq('user_id', user_id)
      .single();
    if (error) {
      console.error("Couldn't remove bookmark:", error);
      setLoading(false);
    } else {
      setBookmarked(false);
      setLoading(false);
    }
  };

  const handleToggle = async () => {
    // if bookmarked, remove bookmark
    if (bookmarked) {
      await removeBookmark();
    } else {
      await addBookmark();
    }
  };

  const baseColor = theme === 'light' ? 'text-gray-600' : 'text-gray-300';
  const hoverColor = theme === 'light' ? 'hover:text-blue-600' : 'hover:text-blue-400';

  const FilledBookmarkIcon = (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path d="M7 3a2 2 0 00-2 2v16l7-4 7 4V5a2 2 0 00-2-2H7z" />
    </svg>
  );

  const OutlineBookmarkIcon = (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="w-5 h-5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-4-7 4V5z" />
    </svg>
  );

  return (
    <button
      onClick={handleToggle}
      disabled={loading}
      className={`flex items-center mt-4 transition-colors duration-200 ${baseColor} ${hoverColor}`}
      aria-label={bookmarked ? 'Remove Bookmark' : 'Add Bookmark'}>
      {bookmarked ? FilledBookmarkIcon : OutlineBookmarkIcon}
      <span className="ml-1 text-sm">{bookmarked ? 'Article has been saved' : 'Bookmark article'}</span>
    </button>
  );
};

export default BookmarkToggle;
