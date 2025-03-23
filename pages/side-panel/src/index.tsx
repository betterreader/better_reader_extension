import { createRoot } from 'react-dom/client';
import { useEffect, useState } from 'react';
import '@src/index.css';
import SidePanel from '@src/SidePanel';
import AuthScreen from '@src/components/AuthScreen';
import { supabase } from '@src/lib/supabase';
import type { Session } from '@supabase/supabase-js';

function App() {
  const [session, setSession] = useState<Session | null>(null);

  async function getSessionFromStorage() {
    try {
      const { session } = (await chrome.storage.local.get('session')) as { session: Session | null };
      if (session) {
        const { error: supaAuthError } = await supabase.auth.setSession(session);
        if (supaAuthError) {
          throw supaAuthError;
        }
        setSession(session);
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
    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => setSession(session));

    // Listen for chrome storage changes
    const storageListener = (changes: { [key: string]: chrome.storage.StorageChange }) => {
      if (changes.session) {
        const newSession = changes.session.newValue;
        if (newSession) {
          // Update Supabase auth state with the new session
          supabase.auth
            .setSession(newSession)
            .then(({ data: { session } }) => {
              setSession(session);
            })
            .catch(error => {
              console.error('Error setting session:', error);
            });
        }
      }
    };

    chrome.storage.local.onChanged.addListener(storageListener);

    // Cleanup subscriptions on unmount
    return () => {
      authListener.subscription.unsubscribe();
      chrome.storage.local.onChanged.removeListener(storageListener);
    };
  }, []);

  return session ? <SidePanel /> : <AuthScreen onLogin={loginWithGoogle} />;
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
