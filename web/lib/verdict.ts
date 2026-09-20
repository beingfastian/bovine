import type { Thresholds } from "./model";
import type { QualityReport } from "./quality";

/**
 * The three-way verdict and the exact words used to render it.
 *
 * These rules are not stylistic. They come from MODEL_CARD.md section 2 and
 * AI_IMPLEMENTATION_PLAN.md section 6.3, and they are non-negotiable:
 *
 *  1. Never render the word "Healthy". The negative verdict is about the
 *     PHOTOGRAPH, not the animal. A false negative delays reporting of a
 *     WOAH-listed notifiable disease.
 *  2. Never name a disease. "Possible skin condition", never "Lumpy Skin
 *     Disease". The user's next action -- show a vet -- is identical either
 *     way, so naming it buys nothing and risks everything.
 *  3. Buffalo sensitivity is entirely unmeasured. A negative result on a
 *     buffalo means even less than a negative result on a cow, and the UI has
 *     to say so.
 */
export type Verdict = "possible_condition" | "unclear" | "no_obvious_lesion";

/** Why we abstained, when we abstained. The user gets told which one. */
export type UnclearReason =
  | "blurry"
  | "dark"
  | "bright"
  | "borderline_score"
  | null;

export interface VerdictResult {
  verdict: Verdict;
  reason: UnclearReason;
  /** Raw model output. Never shown to a farmer as a bare percentage. */
  probability: number;
  concernBand: "low" | "moderate" | "high";
}

export function decide(
  probability: number,
  quality: QualityReport,
  t: Thresholds
): VerdictResult {
  const concernBand =
    probability >= 0.75 ? "high" : probability >= t.tau ? "moderate" : "low";

  // Quality gates run first and short-circuit. A blurry photo's score is not
  // evidence of anything, so it must never become a positive or a negative.
  if (!quality.ok) {
    return { verdict: "unclear", reason: quality.reason, probability, concernBand };
  }

  // Dead-band around tau. Width is measured, not guessed -- see
  // ai/reports/deadband_sweep.json and thresholds.json.
  if (probability >= t.unclearLow && probability < t.unclearHigh) {
    return {
      verdict: "unclear",
      reason: "borderline_score",
      probability,
      concernBand,
    };
  }

  return {
    verdict: probability >= t.tau ? "possible_condition" : "no_obvious_lesion",
    reason: null,
    probability,
    concernBand,
  };
}

interface Copy {
  headline: string;
  headlineUr: string;
  body: string;
  bodyUr: string;
}

export const VERDICT_COPY: Record<Verdict, Copy> = {
  possible_condition: {
    headline: "Possible skin condition",
    headlineUr: "جلد کی ممکنہ بیماری",
    body: "Show this animal to a vet.",
    bodyUr: "جانور کو ڈاکٹر کو دکھائیں۔",
  },
  unclear: {
    headline: "Unclear",
    headlineUr: "واضح نہیں",
    body: "We could not assess this photo reliably. Retake it, or consult a vet if you are concerned.",
    bodyUr: "ہم اس تصویر کا درست جائزہ نہیں لے سکے۔ دوبارہ تصویر لیں، یا فکر ہو تو ڈاکٹر سے رابطہ کریں۔",
  },
  // NOTE: "in this photo" is load-bearing. This says nothing about the animal.
  // The body deliberately avoids the word "healthy" even in negated form --
  // rule 1 is absolute, and "the animal may still be unwell" carries the same
  // meaning more plainly for a reader who is not a native English speaker.
  no_obvious_lesion: {
    headline: "No obvious skin lesion in this photo",
    headlineUr: "اس تصویر میں جلد کا نمایاں مسئلہ نہیں ملا",
    body: "The animal may still be unwell. This reflects one photo, nothing more.",
    bodyUr: "جانور پھر بھی بیمار ہو سکتا ہے۔ یہ صرف ایک تصویر کی بات ہے۔",
  },
};

export const REASON_COPY: Record<NonNullable<UnclearReason>, Copy> = {
  blurry: {
    headline: "The photo is too blurry",
    headlineUr: "تصویر دھندلی ہے",
    body: "Hold the phone still and take it again.",
    bodyUr: "فون کو مستحکم رکھ کر دوبارہ تصویر لیں۔",
  },
  dark: {
    headline: "The photo is too dark",
    headlineUr: "تصویر بہت تاریک ہے",
    body: "Move into better light and take it again.",
    bodyUr: "بہتر روشنی میں جا کر دوبارہ تصویر لیں۔",
  },
  bright: {
    headline: "The photo is washed out",
    headlineUr: "تصویر بہت روشن ہے",
    body: "Move out of direct sun and take it again.",
    bodyUr: "براہِ راست دھوپ سے ہٹ کر دوبارہ تصویر لیں۔",
  },
  borderline_score: {
    headline: "This one is borderline",
    headlineUr: "نتیجہ واضح نہیں",
    body: "Take a closer photo of the skin, or ask a vet.",
    bodyUr: "جلد کی قریب سے تصویر لیں، یا ڈاکٹر سے پوچھیں۔",
  },
};
