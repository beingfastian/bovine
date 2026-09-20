import { BUCKET, getDeviceId, getSupabase } from "./supabase";
import { MODEL_VERSION, type Thresholds } from "./model";
import type { QualityReport } from "./quality";
import type { VerdictResult } from "./verdict";

export interface SubmissionInput {
  image: Blob;
  imageWidth: number;
  imageHeight: number;
  capturedAt: string;
  probability: number;
  latencyMs: number;
  quality: QualityReport;
  verdict: VerdictResult;
  thresholds: Thresholds;
  species: "cattle" | "buffalo";
  lumps: "yes" | "no" | "not_sure";
}

export interface SubmissionResult {
  ok: boolean;
  id?: string;
  error?: string;
}

/**
 * Upload the photo, then insert the record.
 *
 * Deliberately in that order: a row pointing at a missing image is useless,
 * whereas an orphaned image can be reconciled later from the storage listing.
 * If the insert fails we do not try to delete the uploaded object -- anon has
 * no delete permission, by design, so a spent upload is the acceptable cost of
 * not handing strangers the ability to erase collected data.
 *
 * EVERY screening is submitted, including "unclear" ones. Abstentions are
 * data: a rising abstention rate is the earliest signal that the model is
 * drifting away from the field distribution (AI_IMPLEMENTATION_PLAN.md 8.4).
 */
export async function submitScreening(
  input: SubmissionInput
): Promise<SubmissionResult> {
  const pending = getSupabase();
  if (!pending) {
    return { ok: false, error: "Logging is not configured for this build." };
  }
  const supabase = await pending;

  const deviceId = getDeviceId();
  const id = crypto.randomUUID();
  const stamp = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const path = `${stamp}/${id}.jpg`;

  const upload = await supabase.storage.from(BUCKET).upload(path, input.image, {
    contentType: "image/jpeg",
    cacheControl: "3600",
    upsert: false,
  });

  if (upload.error) {
    return { ok: false, error: `Photo upload failed: ${upload.error.message}` };
  }

  const row = {
    id,
    device_id: deviceId,
    captured_at: input.capturedAt,
    image_path: path,
    image_width: input.imageWidth,
    image_height: input.imageHeight,

    model_version: MODEL_VERSION,
    thresholds_version: input.thresholds.thresholdsVersion,
    inference_location: "on-device",

    probability: input.probability,
    verdict: input.verdict.verdict,
    concern_band: input.verdict.concernBand,
    abstain_reason: input.verdict.reason,
    inference_ms: Math.round(input.latencyMs),

    blur_variance: input.quality.blurVariance,
    dark_fraction: input.quality.darkFraction,
    bright_fraction: input.quality.brightFraction,
    mean_luma: input.quality.meanLuma,

    // The labels. These are the point.
    species: input.species,
    reported_lumps: input.lumps,
    // Whether the model has any validated claim for this species at all.
    species_validated: input.species === "cattle",

    user_agent: navigator.userAgent.slice(0, 500),
  };

  const insert = await supabase.from("screenings").insert(row);
  if (insert.error) {
    return { ok: false, error: `Saving the record failed: ${insert.error.message}` };
  }

  return { ok: true, id };
}
