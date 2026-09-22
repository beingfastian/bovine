/**
 * The model contract. Verified against the locked test split -- see
 * ai/reports/onnx_verification.json, ai/reports/gate_report.json and
 * ai/models/bi-lsd-mnv3l-v1.0.0/MODEL_CARD.md.
 *
 * READ BEFORE TOUCHING ANYTHING IN THIS DIRECTORY:
 *
 *   input   : raw [0,255] RGB, 1x224x224x3, NHWC, float32.
 *             The graph rescales internally (MUL 1/127.5, ADD -1).
 *             DO NOT divide by 255. DO NOT apply ImageNet mean/std.
 *   output 1: [1,1] float32 -- ALREADY a sigmoid probability of lesion.
 *             DO NOT apply softmax or sigmoid again.
 *   output 2: [1,3] float32 softmax over [cattle, buffalo, other].
 *             Shares the backbone with output 1; adds one Dense(3) layer.
 *   output 3: [1,1] float32 Mahalanobis d^2 of the pooled features from the
 *             bovine training distribution. Open-set "is this even an
 *             animal photo" -- the softmax alone called landscapes cattle
 *             with p=1.000 (see ai/reports/gate_verification.json).
 *
 *   Outputs are matched by NAME ("lesion", "species", "ood"); two of them
 *   share a shape, so shape alone cannot tell them apart.
 *
 * The previous model was destroyed by exactly this class of mismatch
 * (MODEL_CARD.md section 7). /selftest exists to make a recurrence loud.
 */

/** Lesion weights are bi-lsd-mnv3l-v1.0.0, byte-for-byte. "-gate" marks the
 *  artifact that carries the extra species head, so rows in the database can
 *  tell the two apart. */
export const MODEL_VERSION = "bi-lsd-mnv3l-v1.0.0-gate";

/**
 * float16 weights, fp32 in and out.
 *
 * fp16 measured on the full locked test split (ai/reports/quantization.json):
 * half the download of fp32 for decision agreement 1.0000 and identical
 * balanced accuracy. int8 was built and rejected in the same run: agreement
 * fell to 0.81. MobileNetV3's depthwise convolutions and hard-swish do not
 * survive 8-bit, and the damage is invisible without running the test set.
 */
export const MODEL_URL = `/models/${MODEL_VERSION}.fp16.onnx`;
export const MODEL_PRECISION = "float16";
export const INPUT_SIZE = 224;

/** Graph output names, set by `tf2onnx --rename-outputs` in the export kernel. */
export const OUTPUT_NAMES = { lesion: "lesion", species: "species", ood: "ood" } as const;

/** Order of the species softmax. Fixed by the training script; never reorder. */
export const SPECIES_CLASSES = ["cattle", "buffalo", "other"] as const;
export type SpeciesClass = (typeof SPECIES_CLASSES)[number];

export interface SpeciesProbs {
  cattle: number;
  buffalo: number;
  other: number;
}

/** Runtime-tunable knobs. Fetched from /thresholds.json so they can be retuned
 *  by editing one static file rather than by changing code. */
export interface Thresholds {
  thresholdsVersion: string;
  /** Operating threshold. Selected on the Mild-recall curve, not balanced accuracy. */
  tau: number;
  /** Probabilities inside [unclearLow, unclearHigh) route to "unclear" rather
   *  than to a positive or negative call. Measured, not guessed -- see
   *  ai/reports/onnx_verification.json. */
  unclearLow: number;
  unclearHigh: number;
  /** Gate 1: P(other) at or above this = "we could not find a cattle or
   *  buffalo in this photo". Chosen from the sweep in ai/reports/gate_report.json
   *  so that almost no real animal is turned away -- a wrongly rejected farm
   *  photo is a lost data point, the worst outcome for this app. */
  otherMax: number;
  /** Gate 1, open-set half: Mahalanobis d^2 at or above this = not an animal
   *  photo. Chosen from the OOD sweep in ai/reports/gate_report.json at the
   *  point that keeps >= 99% of real cattle and buffalo. */
  oodMax: number;
  /** Variance of the Laplacian below this = too blurry to assess. */
  blurVarianceMin: number;
  /** Fraction of pixels pinned at 0 or 255 above which the photo is
   *  under/over-exposed enough that we decline to call it. */
  clippedFractionMax: number;
}

/** Mirrors public/thresholds.json, so a failed config fetch lands on the same
 *  operating point rather than a silently different one. */
export const DEFAULT_THRESHOLDS: Thresholds = {
  thresholdsVersion: "2026-09-23b",
  tau: 0.5,
  unclearLow: 0.4,
  unclearHigh: 0.6,
  otherMax: 0.8,
  oodMax: 746.1,
  blurVarianceMin: 120,
  clippedFractionMax: 0.35,
};

let cached: Thresholds | null = null;

export async function loadThresholds(): Promise<Thresholds> {
  if (cached) return cached;
  let next: Thresholds;
  try {
    const res = await fetch("/thresholds.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(String(res.status));
    next = { ...DEFAULT_THRESHOLDS, ...(await res.json()) };
  } catch {
    // A missing or malformed config must not take the app down; fall back to
    // the shipped operating point, which is the one the model card documents.
    next = DEFAULT_THRESHOLDS;
  }
  cached = next;
  return next;
}
