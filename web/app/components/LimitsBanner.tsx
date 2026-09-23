"use client";

import { useState } from "react";

/**
 * The limitations are a product requirement, not a disclaimer to be tucked
 * away. MODEL_CARD.md section 9 approves this model for "collect-only /
 * screening use ... with the section 6 limitations stated in-product".
 *
 * The one-line summary is always visible. The detail expands.
 */
export function LimitsBanner({ species }: { species: "cattle" | "buffalo" | null }) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="rounded-xl border px-4 py-3 text-sm"
      style={{ borderColor: "var(--line)", background: "var(--card)" }}
    >
      <p className="font-medium">
        This is a screening aid, not a diagnosis. It never replaces a vet.
      </p>
      <p className="ur mt-1 text-[15px]" style={{ color: "var(--muted)" }}>
        یہ صرف ابتدائی جانچ ہے، تشخیص نہیں۔ ڈاکٹر کا متبادل نہیں۔
      </p>

      {species === "buffalo" && (
        <p
          className="mt-3 rounded-lg px-3 py-2 text-sm font-medium"
          style={{
            background: "var(--concern-mid-bg)",
            color: "var(--concern-mid)",
          }}
        >
          You marked this animal as a buffalo. The model has never been tested
          on a buffalo with a skin condition, so a negative result here means
          very little. Trust your own eyes and a vet over this result.
        </p>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-2 text-sm underline underline-offset-2"
        style={{ color: "var(--accent)" }}
        aria-expanded={open}
      >
        {open ? "Hide details" : "What this can and cannot do"}
      </button>

      {open && (
        <ul
          className="mt-2 list-disc space-y-1.5 pl-5 text-[13px]"
          style={{ color: "var(--muted)" }}
        >
          <li>
            It was validated on <strong>cattle photographs only</strong>. It has
            never been tested against a photograph taken on a Pakistani farm.
          </li>
          <li>
            <strong>Buffalo are not validated for detection.</strong> We have
            confirmed it does not raise false alarms on buffalo with no skin
            condition, but nobody has ever measured whether it can spot a sick
            one.
          </li>
          <li>
            It is <strong>materially weaker on early or mild cases</strong> than
            on advanced ones.
          </li>
          <li>
            <strong>Most animals it flags will turn out to be fine.</strong> At
            the disease levels typical on a farm, roughly one flagged animal in
            three or four actually has a skin condition. That is how screening
            works — the point is not to miss the one that does. A flag means
            &ldquo;worth a vet&rsquo;s look&rdquo;, nothing more.
          </li>
          <li>
            It looks at one photograph. It knows nothing about the animal&apos;s
            history, temperature, appetite or herd.
          </li>
          <li>
            Before scoring, it checks whether the photo looks like a cattle or
            buffalo at all, and tells you when it does not. That check is new
            and imperfect: if it turns away a real animal, please say so on the
            screen that follows — that is how it gets fixed.
          </li>
        </ul>
      )}
    </div>
  );
}
