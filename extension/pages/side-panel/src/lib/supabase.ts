import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.CEB_SUPABASE_URL;
const supabaseAnonKey = process.env.CEB_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
