import { BUCKET, getDeviceId, getSupabase } from "./supabase";
import { MODEL_VERSION, type SpeciesProbs, type Thresholds } from "./model";
import type { QualityReport } from "./quality";
import type { VerdictResult } from "./verdict";

/**
 * Two writes, at two moments.
 *
 *  saveScreening -- the instant a photo is scored, before the person is asked
 *                   anything. Photo + every number the model produced.
 *  saveLabel     -- when (if) the person answers. A separate INSERT into
 *                   screening_labels, because anonymous visitors have no
 *                   UPDATE right anywhere and that is the whole security
 *                   model. Several labels per screening are allowed; the
 *                   newest wins in the review view.
 *
 * EVERY screening is saved, including "unclear" and "not an animal". Rejected
 * photos are how the gate gets tuned: a real cow turned away is the single
 * worst outcome for a data-collection app, and the only way to see it happen
 * is to keep the evidence.
 */

export interface ScreeningInput {
  id: string;
  image: Blob;
  imageWidth: number;
  imageHeight: number;
  capturedAt: string;
  probability: number;
  species: SpeciesProbs;
  ood: number;
  latencyMs: number;
  quality: QualityReport;
  verdict: VerdictResult;
  thresholds: Thresholds;
}

export interface SaveResult {
  ok: boolean;
  error?: string;
}

/**
 * Insert the record, THEN upload the photo -- in that order.
 *
 * The earlier order (photo first) meant a refused insert -- schema mismatch,
 * or now the rate limit -- left the photo in storage with no row: the
 * invisible kind of orphan, and the one that costs money. Row-first means the
 * rate-limit trigger on `screenings` also bounds storage growth, and
 * migration 003 makes the storage policy refuse any upload whose path has no
 * row. A row whose upload then fails is the visible kind of orphan: /review
 * shows it with a grey box, and it can be retried or deleted.
 */
export async function saveScreening(input: ScreeningInput): Promise<SaveResult> {
  const pending = getSupabase();
  if (!pending) return { ok: false, error: "Saving is not configured for this build." };
  const supabase = await pending;

  const stamp = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const path = `${stamp}/${input.id}.jpg`;

  const row = {
    id: input.id,
    device_id: getDeviceId(),
    captured_at: input.capturedAt,
    image_path: path,
    image_width: input.imageWidth,
    image_height: input.imageHeight,

    model_version: MODEL_VERSION,
    thresholds_version: input.thresholds.thresholdsVersion,
    inference_location: "on-device",
    app_version: process.env.NEXT_PUBLIC_APP_VERSION ?? null,

    probability: input.probability,
    verdict: input.verdict.verdict,
    concern_band: input.verdict.concernBand,
    abstain_reason: input.verdict.reason,
    inference_ms: Math.round(input.latencyMs),

    // Gate 1 / species. Raw probabilities stored so otherMax can be retuned
    // from field data without re-running anything.
    animal_present: input.verdict.animalPresent,
    detected_species: input.verdict.detectedSpecies,
    p_cattle: input.species.cattle,
    p_buffalo: input.species.buffalo,
    p_other: input.species.other,
    ood_distance: input.ood,
    species_validated: input.verdict.detectedSpecies === "cattle",

    blur_variance: input.quality.blurVariance,
    dark_fraction: input.quality.darkFraction,
    bright_fraction: input.quality.brightFraction,
    mean_luma: input.quality.meanLuma,

    user_agent: navigator.userAgent.slice(0, 500),
  };

  const insert = await supabase.from("screenings").insert(row);
  if (insert.error) {
    // The rate-limit trigger raises with this prefix; say something a person
    // can act on rather than echoing an error code.
    const limited = /rate limit/i.test(insert.error.message);
    return {
      ok: false,
      error: limited
        ? "Too many photos saved from this phone in the last hour. The check still works; please try saving again later."
        : `Saving the record failed: ${insert.error.message}`,
    };
  }

  const upload = await supabase.storage.from(BUCKET).upload(path, input.image, {
    contentType: "image/jpeg",
    cacheControl: "3600",
    upsert: false,
  });
  if (upload.error) {
    return { ok: false, error: `Photo upload failed: ${upload.error.message}` };
  }
  return { ok: true };
}

export interface LabelInput {
  screeningId: string;
  species: "cattle" | "buffalo";
  lumps: "yes" | "no" | "not_sure";
  /** True when the person overruled the detector's species. */
  speciesChanged: boolean;
}

export async function saveLabel(input: LabelInput): Promise<SaveResult> {
  const pending = getSupabase();
  if (!pending) return { ok: false, error: "Saving is not configured for this build." };
  const supabase = await pending;

  const insert = await supabase.from("screening_labels").insert({
    screening_id: input.screeningId,
    species: input.species,
    reported_lumps: input.lumps,
    species_changed: input.speciesChanged,
  });
  if (insert.error) {
    return { ok: false, error: `Saving your answer failed: ${insert.error.message}` };
  }
  return { ok: true };
}
