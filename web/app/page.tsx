"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { LimitsBanner } from "./components/LimitsBanner";
import { ResultCard } from "./components/ResultCard";
import { SaveStatus, type SaveState } from "./components/SaveStatus";
import {
  LabelChips,
  type LumpsAnswer,
  type Species,
} from "./components/LabelChips";
import {
  DEFAULT_THRESHOLDS,
  MODEL_VERSION,
  loadThresholds,
  type SpeciesProbs,
  type Thresholds,
} from "@/lib/model";
import { decodeImage, makeUploadJpeg, prepareImage } from "@/lib/preprocess";
import { runInference, warmUp } from "@/lib/infer";
import { assessQuality, type QualityReport } from "@/lib/quality";
import { decide, type VerdictResult } from "@/lib/verdict";
import { saveLabel, saveScreening } from "@/lib/submit";
import { SUPABASE_CONFIGURED } from "@/lib/supabase";

/**
 * Ask for the human label BEFORE revealing the model's verdict.
 *
 * Someone who has just read "possible skin condition" is no longer an
 * independent observer; they are agreeing or disagreeing with a machine. Since
 * the entire purpose of this app is to accumulate trustworthy field labels --
 * above all for diseased buffalo, which exist in no public dataset -- an
 * anchored label would undercut the one thing we are here to collect.
 *
 * Set this to false to reveal the verdict immediately instead.
 */
const ASK_BEFORE_REVEAL = true;

type Stage = "idle" | "working" | "rejected" | "labelling" | "done" | "error";

interface Shot {
  id: string;
  previewUrl: string;
  uploadBlob: Blob;
  width: number;
  height: number;
  probability: number;
  species: SpeciesProbs;
  ood: number;
  latencyMs: number;
  quality: QualityReport;
  verdict: VerdictResult;
  capturedAt: string;
}

export default function Page() {
  const [stage, setStage] = useState<Stage>("idle");
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [shot, setShot] = useState<Shot | null>(null);
  const [species, setSpecies] = useState<Species | null>(null);
  const [lumps, setLumps] = useState<LumpsAnswer | null>(null);
  const [animalConfirmed, setAnimalConfirmed] = useState<boolean | null>(null);
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [modelReady, setModelReady] = useState(false);

  const [photoSave, setPhotoSave] = useState<SaveState>(
    SUPABASE_CONFIGURED ? "idle" : "unconfigured"
  );
  const [answerSave, setAnswerSave] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);

  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  // The label insert has a foreign key to the screening row, so it must wait
  // for the photo save to finish -- on a slow connection the person can
  // easily answer before a 200 kB upload completes.
  const photoSaveRef = useRef<Promise<boolean> | null>(null);
  const lastLabelRef = useRef<string>("");

  // Start downloading the model and config the moment the page opens, so the
  // 6 MB is already in flight while the user is still reading the banner.
  useEffect(() => {
    loadThresholds().then(setThresholds);
    warmUp()
      .then(() => setModelReady(true))
      .catch(() => setModelReady(false));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const startPhotoSave = useCallback(
    (s: Shot, t: Thresholds) => {
      if (!SUPABASE_CONFIGURED) return;
      setPhotoSave("saving");
      setSaveError(null);
      photoSaveRef.current = saveScreening({
        id: s.id,
        image: s.uploadBlob,
        imageWidth: s.width,
        imageHeight: s.height,
        capturedAt: s.capturedAt,
        probability: s.probability,
        species: s.species,
        ood: s.ood,
        latencyMs: s.latencyMs,
        quality: s.quality,
        verdict: s.verdict,
        thresholds: t,
      }).then((r) => {
        setPhotoSave(r.ok ? "saved" : "failed");
        if (!r.ok) setSaveError(r.error ?? "Unknown error");
        return r.ok;
      });
    },
    []
  );

  const startLabelSave = useCallback(
    async (s: Shot, sp: Species, lu: LumpsAnswer) => {
      if (!SUPABASE_CONFIGURED) return;
      const key = `${sp}|${lu}`;
      if (key === lastLabelRef.current) return; // same answer, nothing new to say
      lastLabelRef.current = key;

      setAnswerSave("saving");
      const photoOk = photoSaveRef.current ? await photoSaveRef.current : false;
      if (!photoOk) {
        // No row to attach to. The photo-save failure is already on screen.
        setAnswerSave("failed");
        return;
      }
      const r = await saveLabel({
        screeningId: s.id,
        species: sp,
        lumps: lu,
        speciesChanged:
          s.verdict.detectedSpecies !== "other" &&
          s.verdict.detectedSpecies !== sp,
      });
      setAnswerSave(r.ok ? "saved" : "failed");
      if (!r.ok) setSaveError(r.error ?? "Unknown error");
    },
    []
  );

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setStage("working");
      setSpecies(null);
      setLumps(null);
      setAnimalConfirmed(null);
      setAnswerSave("idle");
      setSaveError(null);
      lastLabelRef.current = "";
      photoSaveRef.current = null;

      try {
        const t = thresholds ?? (await loadThresholds());

        setStatusText("Reading the photo…");
        const bitmap = await decodeImage(file);
        const srcWidth = bitmap.width;
        const srcHeight = bitmap.height;

        setStatusText("Preparing…");
        const prepared = await prepareImage(bitmap);
        const quality = assessQuality(prepared.pixels, t);
        const uploadBlob = await makeUploadJpeg(bitmap, srcWidth, srcHeight);
        bitmap.close();

        setStatusText(modelReady ? "Checking the photo…" : "Loading the model…");
        const inference = await runInference(prepared.tensor);

        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        const previewUrl = URL.createObjectURL(uploadBlob);
        previewUrlRef.current = previewUrl;

        const verdict = decide(
          inference.probability,
          inference.species,
          inference.ood,
          quality,
          t
        );
        const s: Shot = {
          id: crypto.randomUUID(),
          previewUrl,
          uploadBlob,
          width: srcWidth,
          height: srcHeight,
          probability: inference.probability,
          species: inference.species,
          ood: inference.ood,
          latencyMs: inference.latencyMs,
          quality,
          verdict,
          capturedAt: new Date().toISOString(),
        };
        setShot(s);
        setModelReady(true);

        // Save the moment we have a result. Not gated on any answer.
        startPhotoSave(s, t);

        if (verdict.reason === "no_animal") {
          setStage("rejected");
        } else {
          // Pre-fill species from the detector; the person only taps if wrong.
          setSpecies(verdict.detectedSpecies === "buffalo" ? "buffalo" : "cattle");
          setStage(ASK_BEFORE_REVEAL ? "labelling" : "done");
        }
      } catch (e) {
        console.error(e);
        setError(e instanceof Error ? e.message : String(e));
        setStage("error");
      } finally {
        setStatusText("");
      }
    },
    [thresholds, modelReady, startPhotoSave]
  );

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset so picking the same file twice still fires a change event.
    e.target.value = "";
    if (file) void handleFile(file);
  };

  const reset = () => {
    setStage("idle");
    setShot(null);
    setSpecies(null);
    setLumps(null);
    setAnimalConfirmed(null);
    setError(null);
    setPhotoSave(SUPABASE_CONFIGURED ? "idle" : "unconfigured");
    setAnswerSave("idle");
    setSaveError(null);
  };

  const retrySave = () => {
    if (!shot) return;
    const t = thresholds ?? DEFAULT_THRESHOLDS;
    if (photoSave === "failed") startPhotoSave(shot, t);
    if (species && lumps) {
      lastLabelRef.current = "";
      void startLabelSave(shot, species, lumps);
    }
  };

  const reveal = () => {
    if (!shot || !species || !lumps) return;
    setStage("done");
    void startLabelSave(shot, species, lumps);
  };

  // After the reveal, an edited answer is saved again; newest wins.
  const changeSpecies = (sp: Species) => {
    setSpecies(sp);
    if (stage === "done" && shot && lumps) void startLabelSave(shot, sp, lumps);
  };
  const changeLumps = (lu: LumpsAnswer) => {
    setLumps(lu);
    if (stage === "done" && shot && species) void startLabelSave(shot, species, lu);
  };

  return (
    <main className="mx-auto w-full max-w-md px-4 pb-16 pt-6">
      <header className="mb-3">
        <h1 className="text-xl font-semibold tracking-tight">
          BovineInsight — skin check
        </h1>
        <p className="ur text-[16px]" style={{ color: "var(--muted)" }}>
          جانور کی جلد کی جانچ
        </p>
      </header>

      {/* Saving is automatic, so this has to be said up front, every time,
          not buried behind a button that no longer exists. */}
      <p
        className="mb-3 rounded-lg px-3 py-2 text-[12px]"
        style={{ background: "var(--card)", color: "var(--muted)" }}
      >
        Photos you check here are saved to improve the tool. No name, phone
        number or location is stored.
        <span className="ur block text-[13px]">
          یہاں جانچی گئی تصاویر ٹول کو بہتر بنانے کے لیے محفوظ ہوتی ہیں۔ نام، فون
          نمبر یا مقام محفوظ نہیں ہوتا۔
        </span>
      </p>

      <LimitsBanner species={species} />

      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={onPick}
      />
      <input
        ref={galleryRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onPick}
      />

      {stage === "idle" && (
        <section className="mt-5 space-y-3">
          <p className="text-[15px]" style={{ color: "var(--muted)" }}>
            Photograph the animal so its body fills most of the frame, close
            enough to see individual lumps.
          </p>
          <p className="ur text-[15px]" style={{ color: "var(--muted)" }}>
            جانور کی تصویر اس طرح لیں کہ اس کا جسم تصویر کا بڑا حصہ ہو، اتنے
            قریب سے کہ گلٹیاں نظر آئیں۔
          </p>

          <button
            type="button"
            onClick={() => cameraRef.current?.click()}
            className="tap w-full rounded-xl px-5 text-[17px] font-semibold"
            style={{ background: "var(--accent)", color: "var(--bg)" }}
          >
            Take a photo · تصویر لیں
          </button>
          <button
            type="button"
            onClick={() => galleryRef.current?.click()}
            className="tap w-full rounded-xl border px-5 text-[17px] font-medium"
            style={{ borderColor: "var(--line)" }}
          >
            Choose from gallery · گیلری سے چنیں
          </button>

          <p className="text-[12px]" style={{ color: "var(--muted)" }}>
            {modelReady
              ? "Ready."
              : "Loading the model (about 6 MB, once)…"}
          </p>
        </section>
      )}

      {stage === "working" && (
        <section className="mt-6 flex flex-col items-center gap-3 py-10">
          <div
            className="h-8 w-8 animate-spin rounded-full border-2 border-current border-t-transparent"
            style={{ color: "var(--accent)" }}
          />
          <p className="text-[15px]" style={{ color: "var(--muted)" }}>
            {statusText || "Working…"}
          </p>
        </section>
      )}

      {stage === "error" && (
        <section className="mt-6 space-y-3">
          <div
            className="rounded-xl px-4 py-3 text-sm"
            style={{
              background: "var(--concern-high-bg)",
              color: "var(--concern-high)",
            }}
          >
            <p className="font-medium">Something went wrong.</p>
            <p className="mt-1 break-words opacity-90">{error}</p>
          </div>
          <button
            type="button"
            onClick={reset}
            className="tap w-full rounded-xl border px-5 font-medium"
            style={{ borderColor: "var(--line)" }}
          >
            Try again
          </button>
        </section>
      )}

      {shot && (stage === "rejected" || stage === "labelling" || stage === "done") && (
        <section className="mt-5 space-y-5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={shot.previewUrl}
            alt="The photo you captured"
            className="w-full rounded-xl"
            style={{ border: "1px solid var(--line)" }}
          />

          {stage === "rejected" && (
            <>
              <ResultCard result={shot.verdict} />

              {/* A wrongly rejected real animal is the worst outcome for a
                  data-collection app. One tap turns it into a labelled example
                  of the gate being wrong instead of a lost photo. */}
              <div
                className="rounded-xl px-4 py-3"
                style={{ background: "var(--card)" }}
              >
                <p className="text-[15px] font-medium">
                  Is there actually a cattle or buffalo in this photo?
                </p>
                <p className="ur text-[14px]" style={{ color: "var(--muted)" }}>
                  کیا اس تصویر میں واقعی گائے یا بھینس ہے؟
                </p>
                <div className="mt-2 flex gap-2">
                  {(
                    [
                      [true, "Yes", "ہاں"],
                      [false, "No", "نہیں"],
                    ] as [boolean, string, string][]
                  ).map(([v, en, ur]) => (
                    <button
                      key={en}
                      type="button"
                      aria-pressed={animalConfirmed === v}
                      onClick={() => setAnimalConfirmed(v)}
                      className="tap rounded-full border px-5 text-[15px] font-medium"
                      style={{
                        borderColor: animalConfirmed === v ? "var(--accent)" : "var(--line)",
                        background: animalConfirmed === v ? "var(--accent)" : "transparent",
                        color: animalConfirmed === v ? "var(--bg)" : "var(--fg)",
                      }}
                    >
                      {en} <span className="ur ml-2 opacity-80">{ur}</span>
                    </button>
                  ))}
                </div>
              </div>

              {animalConfirmed === true && (
                <>
                  <LabelChips
                    species={species}
                    lumps={lumps}
                    onSpecies={(sp) => {
                      setSpecies(sp);
                      if (lumps) void startLabelSave(shot, sp, lumps);
                    }}
                    onLumps={(lu) => {
                      setLumps(lu);
                      if (species) void startLabelSave(shot, species, lu);
                    }}
                  />
                  <p className="text-[13px]" style={{ color: "var(--muted)" }}>
                    Thank you — this tells us the check was wrong, which is
                    exactly what we need to fix it.
                  </p>
                </>
              )}

              <SaveStatus photo={photoSave} answer={answerSave} error={saveError} onRetry={retrySave} />

              <button
                type="button"
                onClick={reset}
                className="tap w-full rounded-xl px-5 text-[17px] font-semibold"
                style={{ background: "var(--accent)", color: "var(--bg)" }}
              >
                Try another photo · دوبارہ تصویر لیں
              </button>
            </>
          )}

          {stage === "labelling" && (
            <>
              <div
                className="rounded-xl px-4 py-3 text-[15px]"
                style={{ background: "var(--card)" }}
              >
                <p className="font-medium">
                  Before we show the result — what do you see?
                </p>
                <p className="ur mt-1 text-[14px]" style={{ color: "var(--muted)" }}>
                  نتیجہ دکھانے سے پہلے — آپ خود کیا دیکھتے ہیں؟
                </p>
                <p className="mt-2 text-[13px]" style={{ color: "var(--muted)" }}>
                  Your own answer is worth more to us than the model&apos;s, so
                  we ask first.
                </p>
              </div>

              <LabelChips
                species={species}
                lumps={lumps}
                detectedSpecies={
                  shot.verdict.detectedSpecies === "other"
                    ? null
                    : shot.verdict.detectedSpecies
                }
                onSpecies={changeSpecies}
                onLumps={changeLumps}
              />

              <button
                type="button"
                disabled={!species || !lumps}
                onClick={reveal}
                className="tap w-full rounded-xl px-5 text-[17px] font-semibold disabled:opacity-40"
                style={{ background: "var(--accent)", color: "var(--bg)" }}
              >
                Show the result · نتیجہ دکھائیں
              </button>

              <SaveStatus photo={photoSave} answer={answerSave} error={saveError} onRetry={retrySave} />
            </>
          )}

          {stage === "done" && (
            <>
              <ResultCard result={shot.verdict} />

              <LabelChips
                species={species}
                lumps={lumps}
                detectedSpecies={
                  shot.verdict.detectedSpecies === "other"
                    ? null
                    : shot.verdict.detectedSpecies
                }
                onSpecies={changeSpecies}
                onLumps={changeLumps}
              />

              <SaveStatus photo={photoSave} answer={answerSave} error={saveError} onRetry={retrySave} />

              <TechnicalPanel shot={shot} thresholds={thresholds} />

              <button
                type="button"
                onClick={reset}
                className="tap w-full rounded-xl border px-5 font-medium"
                style={{ borderColor: "var(--line)" }}
              >
                Check another animal · دوسرا جانور
              </button>
            </>
          )}
        </section>
      )}
    </main>
  );
}

/**
 * The raw score lives here, behind a tap, clearly marked as a developer view.
 * A bare percentage in the main verdict would be read as diagnostic certainty
 * by exactly the people this is built for.
 */
function TechnicalPanel({
  shot,
  thresholds,
}: {
  shot: Shot;
  thresholds: Thresholds | null;
}) {
  const [open, setOpen] = useState(false);
  const t = thresholds ?? DEFAULT_THRESHOLDS;
  const rows: [string, string][] = [
    ["raw score p(lesion)", shot.probability.toFixed(4)],
    ["threshold tau", t.tau.toFixed(2)],
    ["unclear band", `[${t.unclearLow.toFixed(2)}, ${t.unclearHigh.toFixed(2)})`],
    ["verdict", shot.verdict.verdict],
    ["abstain reason", shot.verdict.reason ?? "—"],
    [
      "species p(cattle/buffalo/other)",
      `${shot.species.cattle.toFixed(3)} / ${shot.species.buffalo.toFixed(3)} / ${shot.species.other.toFixed(3)}`,
    ],
    ["detected", shot.verdict.detectedSpecies],
    ["ood d² (Mahalanobis)", shot.ood.toFixed(1)],
    ["gate oodMax / otherMax", `${t.oodMax >= 1e8 ? "off" : t.oodMax.toFixed(1)} / ${t.otherMax.toFixed(2)}`],
    ["gate failed by", shot.verdict.gateFailedBy.join("+") || "—"],
    ["blur variance", shot.quality.blurVariance.toFixed(1)],
    [
      "dark / bright frac",
      `${shot.quality.darkFraction.toFixed(3)} / ${shot.quality.brightFraction.toFixed(3)}`,
    ],
    ["inference", `${shot.latencyMs.toFixed(0)} ms`],
    ["source image", `${shot.width}×${shot.height}`],
    ["model", MODEL_VERSION],
    ["thresholds", t.thresholdsVersion],
    ["screening id", shot.id.slice(0, 8)],
  ];

  return (
    <div
      className="rounded-xl border"
      style={{ borderColor: "var(--line)", background: "var(--card)" }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="tap w-full px-4 text-left text-[13px] font-medium"
        style={{ color: "var(--muted)" }}
        aria-expanded={open}
      >
        {open ? "▾" : "▸"} Technical details (for testing)
      </button>
      {open && (
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 px-4 pb-4 font-mono text-[12px]">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <dt style={{ color: "var(--muted)" }}>{k}</dt>
              <dd className="text-right">{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
