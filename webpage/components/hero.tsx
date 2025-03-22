import { Button } from './ui/button';
import GoogleSignInButton from './google-sign-in-button';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';
import Link from 'next/link';

export default async function Hero() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] text-center">
      <h1 className="text-6xl font-bold mb-6">Better Reader</h1>
      <p className="text-xl mb-8 max-w-2xl text-muted-foreground">
        Enhance your reading comprehension with AI-powered article analysis, interactive quizzes, and personalized
        learning recommendations.
      </p>
      {user ? (
        <Link href="/protected/analyze">
          <Button size="lg">Start Analyzing</Button>
        </Link>
      ) : (
        <GoogleSignInButton />
      )}
    </div>
  );
}
