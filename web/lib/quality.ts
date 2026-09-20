import type { Thresholds } from "./model";
import type { UnclearReason } from "./verdict";

/**
 * Gate 2 of the screening pipeline (AI_IMPLEMENTATION_PLAN.md section 6.2):
 * is this image usable at all?
 *
 * Gate 1 -- "is there actually a cow or buffalo in frame?" -- needs a detector
 * model and is NOT implemented here. Until it is, this app will happily score a
 * photo of a wall, and the UI has to keep saying so. Do not read the absence of
 * Gate 1 as an oversight; read it as scope.
 *
 * Both measurements run on the same 224x224 pixels the model sees, so the
 * numbers describe the model's actual input rather than the original photo.
 */
export interface QualityReport {
  ok: boolean;
  reason: UnclearReason;
  /** Variance of the Laplacian. Low = out of focus. */
  blurVariance: number;
  /** Fraction of pixels pinned at 0 (crushed) or 255 (blown out). */
  darkFraction: number;
  brightFraction: number;
  meanLuma: number;
}

/** Rec. 601 luma, matching the convention used when these thresholds were
 *  measured over the locked test split. */
function toLuma(px: ImageData): Float32Array {
  const { data, width, height } = px;
  const out = new Float32Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    out[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return out;
}

/** Variance of the 4-neighbour Laplacian -- the standard cheap focus measure. */
function laplacianVariance(luma: Float32Array, w: number, h: number): number {
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v =
        4 * luma[i] - luma[i - 1] - luma[i + 1] - luma[i - w] - luma[i + w];
      sum += v;
      sumSq += v * v;
      n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

export function assessQuality(px: ImageData, t: Thresholds): QualityReport {
  const luma = toLuma(px);
  const blurVariance = laplacianVariance(luma, px.width, px.height);

  let dark = 0;
  let bright = 0;
  let lumaSum = 0;
  for (let i = 0; i < luma.length; i++) {
    lumaSum += luma[i];
    if (luma[i] <= 4) dark++;
    else if (luma[i] >= 251) bright++;
  }
  const darkFraction = dark / luma.length;
  const brightFraction = bright / luma.length;
  const meanLuma = lumaSum / luma.length;

  let reason: UnclearReason = null;
  // Order matters: report the worst problem, and only one, so the retake
  // instruction is a single unambiguous action.
  if (blurVariance < t.blurVarianceMin) reason = "blurry";
  else if (darkFraction > t.clippedFractionMax) reason = "dark";
  else if (brightFraction > t.clippedFractionMax) reason = "bright";

  return {
    ok: reason === null,
    reason,
    blurVariance,
    darkFraction,
    brightFraction,
    meanLuma,
  };
}
