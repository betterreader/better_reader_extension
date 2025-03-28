import { createRoot } from 'react-dom/client';
import { useEffect, useState } from 'react';
import '@src/index.css';
import SidePanel from '@src/SidePanel';
import AuthScreen from '@src/components/AuthScreen';
import { getSupabaseClient } from '@extension/shared/lib/utils/supabaseClient';
import type { Session } from '@supabase/supabase-js';

function App() {
  // TODO: validate if useState is needed
  const [session, setUser] = useState<Session | null>(null);

  // Singleton instance of Supabase client
  const supabase = getSupabaseClient();

  async function getSessionFromStorage() {
    try {
      const { session } = (await chrome.storage.local.get('session')) as { session: Session | null };
      if (session) {
        const { error: supaAuthError } = await supabase.auth.setSession(session);
        if (supaAuthError) {
          throw supaAuthError;
        }
        setUser(session);
      }
    } catch (error) {
      console.error('Error getting session:', error);
    }
  }
  async function loginWithGoogle(): Promise<void> {
    const redirectURL = chrome.identity.getRedirectURL();
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: redirectURL,
      },
    });
    if (error) throw error;

    await chrome.tabs.create({ url: data.url });
  }

  useEffect(() => {
    getSessionFromStorage();

    // Listen for auth state changes to update the session state automatically
    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => setUser(session));

    function handleSessionMessage(message: any) {
      if (message.type === 'SESSION_UPDATED' && 'session' in message) {
        supabase.auth
          .setSession(message.session)
          .then(({ data: { session } }) => {
            setUser(session);
          })
          .catch(error => {
            console.error('Error setting session:', error);
          });
      }
    }

    chrome.runtime.onMessage.addListener(handleSessionMessage);

    // Cleanup subscriptions on unmount
    return () => {
      authListener.subscription.unsubscribe();
      chrome.runtime.onMessage.removeListener(handleSessionMessage);
    };
  }, []);

  return session ? <SidePanel session={session} /> : <AuthScreen onLogin={loginWithGoogle} />;
}

function init() {
  const appContainer = document.querySelector('#app-container');
  if (!appContainer) {
    throw new Error('Cannot find #app-container');
  }
  const root = createRoot(appContainer);
  root.render(<App />);
}

init();
