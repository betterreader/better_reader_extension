import { createClient } from '@/utils/supabase/server';
import { redirect } from 'next/navigation';
import HighlightsClient from './HighlightsClient';
import { Database } from '@/database.types';

type Highlight = Database['public']['Tables']['highlight']['Row'];

export const dynamic = 'force-dynamic';
// (Optional) ensures data is fetched fresh on every request in Next 13

export default async function HighlightsPage() {
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
  const { data: highlights, error } = await supabase
    .from('highlight')
    .select('*')
    .eq('user_id', user.id)
    .order('created_at', { ascending: false });

  // Error state
  if (error) {
    console.error('Error fetching highlights:', error);
    return (
      <div className="container mx-auto py-8">
        <div className="text-center p-8 rounded-lg">
          <h2 className="text-xl font-semibold text-red-700 mb-2">Error Loading Highlights</h2>
          <p className="text-red-600">There was a problem loading your highlights. Please try again later.</p>
        </div>
      </div>
    );
  }

  // Empty state
  if (!highlights || highlights.length === 0) {
    return (
      <div className="container mx-auto py-8">
        <div className="text-center p-8 rounded-lg">
          <h2 className="text-xl font-semibold mb-2">No Highlights Found</h2>
          <p className="text-sm">Start highlighting text on articles to see them appear here.</p>
        </div>
      </div>
    );
  }

  // Group highlights by URL
  const grouped = highlights.reduce<
    Record<
      string,
      {
        url: string;
        title: string;
        highlights: Highlight[];
      }
    >
  >((acc, highlight) => {
    const key = highlight.url;
    if (!acc[key]) {
      acc[key] = {
        url: highlight.url,
        title: highlight.article_title,
        highlights: [],
      };
    }
    acc[key].highlights.push(highlight);
    return acc;
  }, {});

  // Convert grouped object into array for easier mapping in the client
  const groupedHighlights = Object.values(grouped);

  return <HighlightsClient groupedHighlights={groupedHighlights} />;
}
