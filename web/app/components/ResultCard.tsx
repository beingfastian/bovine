"use client";

import {
  REASON_COPY,
  VERDICT_COPY,
  type VerdictResult,
} from "@/lib/verdict";

const BAND_STYLE = {
  high: { fg: "var(--concern-high)", bg: "var(--concern-high-bg)" },
  moderate: { fg: "var(--concern-mid)", bg: "var(--concern-mid-bg)" },
  low: { fg: "var(--concern-low)", bg: "var(--concern-low-bg)" },
} as const;

const UNCLEAR_STYLE = { fg: "var(--unclear)", bg: "var(--unclear-bg)" };

export function ResultCard({ result }: { result: VerdictResult }) {
  const unclear = result.verdict === "unclear";
  const style = unclear ? UNCLEAR_STYLE : BAND_STYLE[result.concernBand];

  // When we abstain, the reason IS the message -- "retake, it's too dark" is
  // actionable in a way that "unclear" alone is not.
  const copy =
    unclear && result.reason
      ? REASON_COPY[result.reason]
      : VERDICT_COPY[result.verdict];

  return (
    <div
      className="rounded-2xl px-5 py-5"
      style={{ background: style.bg, color: style.fg }}
      role="status"
      aria-live="polite"
    >
      <p className="text-[22px] font-semibold leading-tight">{copy.headline}</p>
      <p className="ur mt-1 text-[19px] font-semibold">{copy.headlineUr}</p>

      <p className="mt-3 text-[15px] leading-snug opacity-90">{copy.body}</p>
      <p className="ur mt-1 text-[15px] opacity-90">{copy.bodyUr}</p>

      {unclear && result.reason === "borderline_score" && (
        <p className="mt-3 text-[13px] opacity-80">
          The score landed too close to the cut-off to call it either way.
        </p>
      )}

      {result.verdict === "no_obvious_lesion" && (
        <p className="mt-3 text-[13px] opacity-80">
          This is about the photograph, not the animal. If it seems unwell,
          see a vet regardless of what this says.
        </p>
      )}

      <ConcernMeter band={result.concernBand} unclear={unclear} />
    </div>
  );
}

/**
 * A three-band indicator rather than a number.
 * AI_IMPLEMENTATION_PLAN.md section 6.3 rule 4: a farmer reading "73%" will
 * read it as diagnostic certainty, which it is not. The raw score still exists
 * -- it lives in the technical panel, for the person testing the model.
 */
function ConcernMeter({
  band,
  unclear,
}: {
  band: "low" | "moderate" | "high";
  unclear: boolean;
}) {
  const steps = ["low", "moderate", "high"] as const;
  const activeIndex = unclear ? -1 : steps.indexOf(band);

  return (
    <div className="mt-4">
      <div className="flex gap-1.5" aria-hidden>
        {steps.map((s, i) => (
          <div
            key={s}
            className="h-2 flex-1 rounded-full"
            style={{
              background: "currentColor",
              opacity: i <= activeIndex ? 0.85 : 0.18,
            }}
          />
        ))}
      </div>
      <p className="mt-1.5 text-[12px] uppercase tracking-wide opacity-75">
        {unclear ? "not assessed" : `${band} concern`}
      </p>
    </div>
  );
}
