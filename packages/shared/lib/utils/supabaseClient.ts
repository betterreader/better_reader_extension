// lib/supabaseClient.ts
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { Database } from '../../types/supabase.js';

const SUPABASE_URL = process.env.CEB_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.CEB_SUPABASE_ANON_KEY!;

// Singleton instance
let supabase: SupabaseClient | null = null;

export function getSupabaseClient(): SupabaseClient<Database> {
  if (!supabase) {
    supabase = createClient<Database>(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        persistSession: false, // Extensions don't support localStorage in background/popup
        autoRefreshToken: false, // You handle refresh manually or via setSession
      },
    });
  }

  return supabase;
}
