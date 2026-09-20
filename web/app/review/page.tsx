"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { SupabaseClient } from "@supabase/supabase-js";
import { BUCKET, SUPABASE_CONFIGURED, getSupabase } from "@/lib/supabase";

/**
 * Submission review.
 *
 * Behind a Supabase magic-link sign-in, because the anon key is public: the
 * insert-only RLS policy means anonymous visitors cannot read this table even
 * though they can write to it. Photos come back as short-lived signed URLs
 * rather than public links, so the bucket never has to be made public.
 *
 * The summary at the top is the number that matters. Cattle accuracy is
 * already measured; what nobody has is buffalo with confirmed lesions.
 */

interface Row {
  id: string;
  created_at: string;
  captured_at: string | null;
  device_id: string | null;
  image_path: string;
  probability: number;
  verdict: string;
  concern_band: string | null;
  abstain_reason: string | null;
  inference_ms: number | null;
  blur_variance: number | null;
  species: "cattle" | "buffalo";
  reported_lumps: "yes" | "no" | "not_sure";
  model_version: string;
  vet_verdict: string | null;
  reviewer_note: string | null;
}

type Filter = "all" | "buffalo_lumps" | "disagreements" | "unclear";

export default function ReviewPage() {
  const [supabase, setSupabase] = useState<SupabaseClient | null>(null);
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [signedIn, setSignedIn] = useState<boolean | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Resolve the lazily-imported client once, then watch the session.
  useEffect(() => {
    const pending = getSupabase();
    if (!pending) {
      setSignedIn(false);
      return;
    }
    let unsubscribe: (() => void) | undefined;
    let cancelled = false;
    void pending.then(async (client) => {
      if (cancelled) return;
      setSupabase(client);
      const { data } = await client.auth.getSession();
      setSignedIn(!!data.session);
      const { data: sub } = client.auth.onAuthStateChange((_e, session) =>
        setSignedIn(!!session)
      );
      unsubscribe = () => sub.subscription.unsubscribe();
    });
    return () => {
      cancelled = true;
      unsubscribe?.();
    };
  }, []);

  const load = useCallback(async () => {
    if (!supabase) return;
    setLoading(true);
    setError(null);

    const { data, error: err } = await supabase
      .from("screenings")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(300);

    if (err) {
      setError(err.message);
      setLoading(false);
      return;
    }

    const list = (data ?? []) as Row[];
    setRows(list);

    // Signed URLs, in one batch rather than one request per photo.
    const paths = list.map((r) => r.image_path);
    if (paths.length) {
      const { data: signed } = await supabase.storage
        .from(BUCKET)
        .createSignedUrls(paths, 3600);
      const map: Record<string, string> = {};
      for (const s of signed ?? []) {
        if (s.signedUrl && s.path) map[s.path] = s.signedUrl;
      }
      setUrls(map);
    }
    setLoading(false);
  }, [supabase]);

  useEffect(() => {
    if (signedIn) void load();
  }, [signedIn, load]);

  const summary = useMemo(() => {
    const cell = (sp: string, lu: string) =>
      rows.filter((r) => r.species === sp && r.reported_lumps === lu).length;
    return {
      total: rows.length,
      cattleYes: cell("cattle", "yes"),
      cattleNo: cell("cattle", "no"),
      buffaloYes: cell("buffalo", "yes"),
      buffaloNo: cell("buffalo", "no"),
      notSure: rows.filter((r) => r.reported_lumps === "not_sure").length,
      unclear: rows.filter((r) => r.verdict === "unclear").length,
    };
  }, [rows]);

  const visible = useMemo(() => {
    switch (filter) {
      case "buffalo_lumps":
        return rows.filter(
          (r) => r.species === "buffalo" && r.reported_lumps === "yes"
        );
      case "disagreements":
        // Where the person and the model actually contradict each other.
        return rows.filter(
          (r) =>
            (r.reported_lumps === "yes" && r.verdict === "no_obvious_lesion") ||
            (r.reported_lumps === "no" && r.verdict === "possible_condition")
        );
      case "unclear":
        return rows.filter((r) => r.verdict === "unclear");
      default:
        return rows;
    }
  }, [rows, filter]);

  const exportCsv = () => {
    const cols = [
      "id", "created_at", "species", "reported_lumps", "verdict",
      "probability", "concern_band", "abstain_reason", "blur_variance",
      "inference_ms", "model_version", "device_id", "image_path",
      "vet_verdict", "reviewer_note",
    ] as const;
    const esc = (v: unknown) =>
      `"${String(v ?? "").replace(/"/g, '""')}"`;
    const csv = [
      cols.join(","),
      ...visible.map((r) => cols.map((c) => esc(r[c as keyof Row])).join(",")),
    ].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `bovineinsight-${filter}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!SUPABASE_CONFIGURED) {
    return (
      <main className="mx-auto max-w-2xl px-4 py-10">
        <h1 className="text-xl font-semibold">Review</h1>
        <p className="mt-3 text-sm" style={{ color: "var(--muted)" }}>
          Supabase is not configured for this build. Set
          NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY, then
          rebuild.
        </p>
      </main>
    );
  }

  if (signedIn === false) {
    return (
      <main className="mx-auto max-w-sm px-4 py-12">
        <h1 className="text-xl font-semibold">Review — sign in</h1>
        <p className="mt-2 text-sm" style={{ color: "var(--muted)" }}>
          Submissions are readable only by a signed-in account.
        </p>
        {sent ? (
          <p className="mt-6 text-sm">
            Check your email for the sign-in link, then return to this page.
          </p>
        ) : (
          <form
            className="mt-6 space-y-3"
            onSubmit={async (e) => {
              e.preventDefault();
              const { error: err } = await supabase!.auth.signInWithOtp({
                email,
                options: { emailRedirectTo: window.location.href },
              });
              if (err) setError(err.message);
              else setSent(true);
            }}
          >
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="tap w-full rounded-xl border px-4"
              style={{ borderColor: "var(--line)", background: "var(--card)" }}
            />
            <button
              type="submit"
              className="tap w-full rounded-xl px-5 font-semibold"
              style={{ background: "var(--accent)", color: "var(--bg)" }}
            >
              Email me a sign-in link
            </button>
          </form>
        )}
        {error && (
          <p className="mt-3 text-sm" style={{ color: "var(--concern-high)" }}>
            {error}
          </p>
        )}
      </main>
    );
  }

  if (signedIn === null) {
    return (
      <main className="mx-auto max-w-2xl px-4 py-10">
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Checking session…
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-3xl px-4 pb-16 pt-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-semibold">Submissions</h1>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => void load()}
            className="rounded-lg border px-3 py-1.5 text-sm"
            style={{ borderColor: "var(--line)" }}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
          <button
            type="button"
            onClick={exportCsv}
            disabled={!visible.length}
            className="rounded-lg border px-3 py-1.5 text-sm disabled:opacity-40"
            style={{ borderColor: "var(--line)" }}
          >
            Export CSV
          </button>
          <button
            type="button"
            onClick={() => void supabase!.auth.signOut()}
            className="rounded-lg border px-3 py-1.5 text-sm"
            style={{ borderColor: "var(--line)" }}
          >
            Sign out
          </button>
        </div>
      </div>

      <div
        className="mt-4 grid grid-cols-2 gap-3 rounded-xl border p-4 sm:grid-cols-3"
        style={{ borderColor: "var(--line)", background: "var(--card)" }}
      >
        <Stat label="total" value={summary.total} />
        <Stat label="cattle · lumps" value={summary.cattleYes} />
        <Stat label="cattle · no lumps" value={summary.cattleNo} />
        <Stat
          label="BUFFALO · lumps"
          value={summary.buffaloYes}
          highlight
          note="the gap the project exists to close"
        />
        <Stat label="buffalo · no lumps" value={summary.buffaloNo} />
        <Stat label="model said unclear" value={summary.unclear} />
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {(
          [
            ["all", "All"],
            ["buffalo_lumps", "Buffalo with lumps"],
            ["disagreements", "Person ≠ model"],
            ["unclear", "Unclear"],
          ] as [Filter, string][]
        ).map(([k, label]) => (
          <button
            key={k}
            type="button"
            onClick={() => setFilter(k)}
            className="rounded-full border px-4 py-1.5 text-sm"
            style={{
              borderColor: filter === k ? "var(--accent)" : "var(--line)",
              background: filter === k ? "var(--accent)" : "transparent",
              color: filter === k ? "var(--bg)" : "var(--fg)",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {error && (
        <p className="mt-4 text-sm" style={{ color: "var(--concern-high)" }}>
          {error}
        </p>
      )}

      {!loading && !visible.length && (
        <p className="mt-8 text-sm" style={{ color: "var(--muted)" }}>
          Nothing here yet.
        </p>
      )}

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        {visible.map((r) => (
          <Card key={r.id} row={r} url={urls[r.image_path]} />
        ))}
      </div>
    </main>
  );
}

function Stat({
  label,
  value,
  highlight,
  note,
}: {
  label: string;
  value: number;
  highlight?: boolean;
  note?: string;
}) {
  return (
    <div>
      <p
        className="text-2xl font-semibold"
        style={{ color: highlight ? "var(--accent)" : undefined }}
      >
        {value}
      </p>
      <p className="text-[12px] uppercase tracking-wide" style={{ color: "var(--muted)" }}>
        {label}
      </p>
      {note && (
        <p className="text-[11px]" style={{ color: "var(--muted)" }}>
          {note}
        </p>
      )}
    </div>
  );
}

function Card({ row, url }: { row: Row; url?: string }) {
  const disagrees =
    (row.reported_lumps === "yes" && row.verdict === "no_obvious_lesion") ||
    (row.reported_lumps === "no" && row.verdict === "possible_condition");

  return (
    <div
      className="overflow-hidden rounded-xl border"
      style={{ borderColor: "var(--line)" }}
    >
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <a href={url} target="_blank" rel="noreferrer">
          <img src={url} alt="" className="aspect-square w-full object-cover" />
        </a>
      ) : (
        <div
          className="aspect-square w-full"
          style={{ background: "var(--card)" }}
        />
      )}
      <div className="space-y-1 px-3 py-2 text-[12px]">
        <p className="font-medium">
          {row.species} · reported lumps: {row.reported_lumps}
        </p>
        <p style={{ color: "var(--muted)" }}>
          model: {row.verdict} ({row.probability.toFixed(3)})
          {row.abstain_reason ? ` · ${row.abstain_reason}` : ""}
        </p>
        {disagrees && (
          <p style={{ color: "var(--concern-mid)" }}>person disagrees with model</p>
        )}
        <p className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
          {new Date(row.created_at).toLocaleString()} ·{" "}
          {row.device_id?.slice(0, 8) ?? "?"}
        </p>
      </div>
    </div>
  );
}
