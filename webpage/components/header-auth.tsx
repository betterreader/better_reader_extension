import { signOutAction } from '@/app/actions';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { createClient } from '@/utils/supabase/server';
import GoogleSignInButton from './google-sign-in-button';
import Link from 'next/link';
import { ThemeSwitcher } from './theme-switcher';

export default async function AuthButton() {
  const supabase = await createClient();

  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <div className="flex items-center gap-4">
      <div className="w-10 h-10 flex items-center justify-center">
        <ThemeSwitcher />
      </div>
      <Link href="/">Home</Link>
      {user ? (
        <>
          <Link href="/protected/highlights">Highlights</Link>
          <Link href="/protected/bookmarks">Bookmarks</Link>
          <form action={signOutAction}>
            <Button type="submit" variant={'outline'}>
              Sign out
            </Button>
          </form>
        </>
      ) : (
        <GoogleSignInButton />
      )}
    </div>
  );
}
