'use client';

import { Button } from './ui/button';
import { createClient } from '@/utils/supabase/client';

export default function GoogleSignInButton() {
  const supabase = createClient();

  const handleSignIn = async () => {
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });
  };

  return (
    <Button onClick={handleSignIn} size="sm" variant={'default'}>
      Sign in with Google
    </Button>
  );
}
