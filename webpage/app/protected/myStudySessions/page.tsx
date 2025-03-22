'use client';

import { createClient } from '@/utils/supabase/client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface StudySession {
  id: number;
  url: string;
  created_at: string;
  title: string;
}

export default function MyStudySessionsPage() {
  const [sessions, setSessions] = useState<StudySession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const supabase = createClient();
        const { data, error } = await supabase
          .from('analysis')
          .select('id, url, created_at, title')
          .order('created_at', { ascending: false });

        if (error) throw error;

        setSessions(data || []);
      } catch (err) {
        console.error('Error fetching study sessions:', err);
        setError('Failed to load study sessions');
      } finally {
        setLoading(false);
      }
    };

    fetchSessions();
  }, []);

  const handleSessionClick = (id: number) => {
    router.push(`/protected/study?id=${id}`);
  };

  if (loading) {
    return (
      <main className="container mx-auto px-4 py-8 max-w-full sm:max-w-4xl">
        <h1 className="text-2xl md:text-4xl font-bold mb-8 text-center">My Study Sessions</h1>
        <p className="text-center">Loading study sessions...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="container mx-auto px-4 py-8 max-w-full sm:max-w-4xl">
        <h1 className="text-2xl md:text-4xl font-bold mb-8 text-center">My Study Sessions</h1>
        <p className="text-center text-red-500">{error}</p>
      </main>
    );
  }

  return (
    <main className="container mx-auto px-4 py-8 max-w-full sm:max-w-4xl">
      <h1 className="text-2xl md:text-4xl font-bold mb-8 text-center">My Study Sessions</h1>

      {sessions.length === 0 ? (
        <div className="text-center">
          <p className="mb-4">You haven't analyzed any articles yet.</p>
          <Button onClick={() => router.push('/protected/analyze')}>Analyze an Article</Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {sessions.map(session => (
            <Card
              key={session.id}
              className="p-6 cursor-pointer hover:bg-accent transition-colors"
              onClick={() => handleSessionClick(session.id)}>
              <div className="space-y-2">
                <h2 className="text-lg md:text-xl font-semibold line-clamp-2">{session.title}</h2>
                <p className="text-xs md:text-sm text-muted-foreground truncate">{session.url}</p>
                <p className="text-xs md:text-sm text-muted-foreground">
                  {new Date(session.created_at).toLocaleDateString()}
                </p>
              </div>
            </Card>
          ))}
        </div>
      )}
    </main>
  );
}
