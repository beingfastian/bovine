import type { SupabaseClient } from "@supabase/supabase-js";

/**
 * Supabase access for a fully static site.
 *
 * The anon key is compiled into the bundle and is meant to be public -- it is
 * an identity, not a secret. Everything that protects the data is a Row Level
 * Security policy in supabase/schema.sql: anonymous visitors may INSERT a
 * screening and upload one image, and may read nothing at all. Reading is
 * reserved for a signed-in account, which is what /review uses.
 *
 * The service_role key must NEVER appear in this directory.
 *
 * supabase-js is imported dynamically. It is ~65 kB that the capture screen
 * does not need until someone actually presses Send, and this app opens on a
 * connection where that is real waiting time.
 */

const URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const ANON = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export const SUPABASE_CONFIGURED = Boolean(URL && ANON);

export const BUCKET = "screenings";

let clientPromise: Promise<SupabaseClient> | null = null;

export function getSupabase(): Promise<SupabaseClient> | null {
  if (!SUPABASE_CONFIGURED) return null;
  if (!clientPromise) {
    clientPromise = import("@supabase/supabase-js").then(({ createClient }) =>
      createClient(URL!, ANON!, {
        auth: { persistSession: true, autoRefreshToken: true },
      })
    );
  }
  return clientPromise;
}

/**
 * A stable anonymous id for this device.
 *
 * Not a login and not a tracker: it exists so that twenty photos from one farm
 * can be recognised as twenty photos from one farm during analysis, which
 * matters enormously when judging whether the field set is genuinely diverse
 * or is one shed photographed repeatedly.
 */
export function getDeviceId(): string {
  const KEY = "bi_device_id";
  try {
    const existing = localStorage.getItem(KEY);
    if (existing) return existing;
    const id = crypto.randomUUID();
    localStorage.setItem(KEY, id);
    return id;
  } catch {
    // Private mode, or storage blocked. A per-load id is still better than
    // nothing; it just cannot be grouped across visits.
    return crypto.randomUUID();
  }
}
