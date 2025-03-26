// lib/supabaseClient.ts
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { Database } from '../../types/supabase.js';

const SUPABASE_URL = process.env.CEB_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.CEB_SUPABASE_ANON_KEY!;

// Singleton instance
let supabase: SupabaseClient | null = null;

export function getSupabaseClient(): SupabaseClient {
  if (!supabase) {
    supabase = createClient(
      'http://127.0.0.1:54321',
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0',
      {
        auth: {
          persistSession: false, // Extensions don't support localStorage in background/popup
          autoRefreshToken: false, // You handle refresh manually or via setSession
        },
      },
    );
  }

  return supabase;
}
