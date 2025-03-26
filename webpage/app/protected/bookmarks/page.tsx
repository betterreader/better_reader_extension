import { createClient } from '@/utils/supabase/server';
import { redirect } from 'next/navigation';
import BookmarkClient from './BookmarkClient';
import { Database } from '@/database.types';

type Bookmark = Database['public']['Tables']['bookmark']['Row'];

export const dynamic = 'force-dynamic';
// (Optional) ensures data is fetched fresh on every request in Next 13

export default async function BookmarkPage() {
  const supabase = await createClient();

  // Get current user
  const {
    data: { user },
  } = await supabase.auth.getUser();

  // Redirect if not logged in
  if (!user) {
    redirect('/auth/sign-in');
  }

  // Fetch highlights for this user
  const { data: bookmarks, error } = await supabase
    .from('bookmark')
    .select('*')
    .eq('user_id', user.id)
    .order('created_at', { ascending: false });

  // Error state
  if (error) {
    console.error('Error fetching bookmarks:', error);
    return (
      <div className="container mx-auto py-8">
        <div className="text-center p-8 rounded-lg">
          <h2 className="text-xl font-semibold text-red-700 mb-2">Error Loading Bookmarks</h2>
          <p className="text-red-600">There was a problem loading your bookmarks. Please try again later.</p>
        </div>
      </div>
    );
  }

  // Empty state
  if (!bookmarks || bookmarks.length === 0) {
    return (
      <div className="container mx-auto py-8">
        <div className="text-center p-8 rounded-lg">
          <h2 className="text-xl font-semibold mb-2">No Bookmarks Found</h2>
          <p className="text-sm">Start bookmarking articles to see them appear here.</p>
        </div>
      </div>
    );
  }

  return <BookmarkClient bookmarks={bookmarks} />;
}
