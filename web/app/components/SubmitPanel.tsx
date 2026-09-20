"use client";

import { useState } from "react";
import { SUPABASE_CONFIGURED } from "@/lib/supabase";
import { submitScreening, type SubmissionInput } from "@/lib/submit";

type State = "idle" | "sending" | "sent" | "failed";

/**
 * Sending is an explicit act, and the screen says plainly what it does before
 * the person does it.
 *
 * AI_IMPLEMENTATION_PLAN.md section 11 keeps "keep my photo" and "train on my
 * photo" as separate permissions and defaults the second to off. This app has
 * only one purpose -- building the field set -- so there is nothing to
 * separate: pressing the button IS the consent, and the button therefore has
 * to describe what it is consenting to, in both languages, before it is
 * pressed.
 */
export function SubmitPanel({ payload }: { payload: SubmissionInput }) {
  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState<string | null>(null);

  if (!SUPABASE_CONFIGURED) {
    return (
      <p
        className="rounded-xl px-4 py-3 text-[13px]"
        style={{ background: "var(--card)", color: "var(--muted)" }}
      >
        Data collection is not configured for this build, so nothing is being
        sent anywhere. Set NEXT_PUBLIC_SUPABASE_URL and
        NEXT_PUBLIC_SUPABASE_ANON_KEY to enable it.
      </p>
    );
  }

  if (state === "sent") {
    return (
      <div
        className="rounded-xl px-4 py-3"
        style={{
          background: "var(--concern-low-bg)",
          color: "var(--concern-low)",
        }}
      >
        <p className="text-[15px] font-medium">Sent — thank you.</p>
        <p className="ur mt-1 text-[15px]">بھیج دیا گیا — شکریہ۔</p>
        <p className="mt-2 text-[13px] opacity-85">
          Photos like this one, especially of buffalo, are what make the tool
          better.
        </p>
      </div>
    );
  }

  const send = async () => {
    setState("sending");
    setError(null);
    const res = await submitScreening(payload);
    if (res.ok) {
      setState("sent");
    } else {
      setError(res.error ?? "Unknown error");
      setState("failed");
    }
  };

  return (
    <div
      className="rounded-xl border px-4 py-3"
      style={{ borderColor: "var(--line)", background: "var(--card)" }}
    >
      <p className="text-[15px] font-medium">Help improve this tool</p>
      <p className="ur text-[15px]" style={{ color: "var(--muted)" }}>
        اس ٹول کو بہتر بنانے میں مدد کریں
      </p>
      <p className="mt-2 text-[13px]" style={{ color: "var(--muted)" }}>
        Sending stores this photo and your two answers so the tool can be
        improved. No name, no phone number, no location.
      </p>
      <p className="ur mt-1 text-[14px]" style={{ color: "var(--muted)" }}>
        بھیجنے سے یہ تصویر اور آپ کے دو جواب محفوظ ہوں گے۔ نام، فون نمبر یا
        مقام محفوظ نہیں ہوتا۔
      </p>

      <button
        type="button"
        onClick={() => void send()}
        disabled={state === "sending"}
        className="tap mt-3 w-full rounded-xl px-5 text-[16px] font-semibold disabled:opacity-50"
        style={{ background: "var(--accent)", color: "var(--bg)" }}
      >
        {state === "sending" ? "Sending…" : "Send this photo · تصویر بھیجیں"}
      </button>

      {state === "failed" && (
        <div className="mt-2">
          <p className="text-[13px]" style={{ color: "var(--concern-high)" }}>
            {error}
          </p>
          <button
            type="button"
            onClick={() => void send()}
            className="tap mt-1 text-[13px] underline"
            style={{ color: "var(--accent)" }}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
