/**
 * The model contract. Verified against the locked test split -- see
 * ai/reports/onnx_export_report.json and ai/models/bi-lsd-mnv3l-v1.0.0/MODEL_CARD.md.
 *
 * READ BEFORE TOUCHING ANYTHING IN THIS DIRECTORY:
 *
 *   input  : raw [0,255] RGB, 1x224x224x3, NHWC, float32.
 *            The graph rescales internally (MUL 1/127.5, ADD -1).
 *            DO NOT divide by 255. DO NOT apply ImageNet mean/std.
 *   output : [1,1] float32 -- ALREADY a sigmoid probability of lesion.
 *            DO NOT apply softmax or sigmoid again.
 *
 * The previous model was destroyed by exactly this class of mismatch
 * (MODEL_CARD.md section 7). lib/selftest.ts exists to make a recurrence loud.
 */

export const MODEL_VERSION = "bi-lsd-mnv3l-v1.0.0";

/**
 * float16 weights, fp32 in and out.
 *
 * Measured on the full locked test split (ai/reports/quantization.json):
 * 5.73 MB gzipped against 11.53 MB for fp32 -- half the download on a
 * 600 kB/s connection -- for decision agreement 1.0000 and balanced accuracy
 * identical to four decimal places. Max probability difference 0.0088, the
 * same order as the fp16 TFLite already shipped on Android.
 *
 * int8 was built and rejected in the same run: 2.92 MB, but agreement fell to
 * 0.8101 and balanced accuracy to 0.7048. MobileNetV3's depthwise convolutions
 * and hard-swish activations do not survive 8-bit quantisation, and the damage
 * is invisible without running the test set.
 */
export const MODEL_URL = `/models/${MODEL_VERSION}.fp16.onnx`;
export const MODEL_PRECISION = "float16";
export const INPUT_SIZE = 224;

/** Runtime-tunable knobs. Fetched from /thresholds.json so they can be retuned
 *  by editing one static file rather than by changing code. */
export interface Thresholds {
  thresholdsVersion: string;
  /** Operating threshold. Selected on the Mild-recall curve, not balanced accuracy. */
  tau: number;
  /** Probabilities inside [unclearLow, unclearHigh) route to "unclear" rather
   *  than to a positive or negative call. Measured, not guessed -- see
   *  ai/reports/deadband_sweep.json. */
  unclearLow: number;
  unclearHigh: number;
  /** Variance of the Laplacian below this = too blurry to assess. */
  blurVarianceMin: number;
  /** Fraction of pixels pinned at 0 or 255 above which the photo is
   *  under/over-exposed enough that we decline to call it. */
  clippedFractionMax: number;
}

/** Mirrors public/thresholds.json, so a failed config fetch lands on the same
 *  operating point rather than a silently different one. */
export const DEFAULT_THRESHOLDS: Thresholds = {
  thresholdsVersion: "2026-09-21",
  tau: 0.5,
  unclearLow: 0.4,
  unclearHigh: 0.6,
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
