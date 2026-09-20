"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { LimitsBanner } from "./components/LimitsBanner";
import { ResultCard } from "./components/ResultCard";
import { SubmitPanel } from "./components/SubmitPanel";
import {
  LabelChips,
  type LumpsAnswer,
  type Species,
} from "./components/LabelChips";
import {
  DEFAULT_THRESHOLDS,
  MODEL_VERSION,
  loadThresholds,
  type Thresholds,
} from "@/lib/model";
import { decodeImage, makeUploadJpeg, prepareImage } from "@/lib/preprocess";
import { runInference, warmUp } from "@/lib/infer";
import { assessQuality, type QualityReport } from "@/lib/quality";
import { decide, type VerdictResult } from "@/lib/verdict";

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

type Stage = "idle" | "working" | "labelling" | "done" | "error";

interface Shot {
  previewUrl: string;
  uploadBlob: Blob;
  width: number;
  height: number;
  probability: number;
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
  const [thresholds, setThresholds] = useState<Thresholds | null>(null);
  const [modelReady, setModelReady] = useState(false);

  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  // Start downloading the model and config the moment the page opens, so the
  // 12 MB is already in flight while the user is still reading the banner.
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

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setStage("working");
      setSpecies(null);
      setLumps(null);

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

        setStatusText(modelReady ? "Checking the skin…" : "Loading the model…");
        const inference = await runInference(prepared.tensor);

        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        const previewUrl = URL.createObjectURL(uploadBlob);
        previewUrlRef.current = previewUrl;

        setShot({
          previewUrl,
          uploadBlob,
          width: srcWidth,
          height: srcHeight,
          probability: inference.probability,
          latencyMs: inference.latencyMs,
          quality,
          verdict: decide(inference.probability, quality, t),
          capturedAt: new Date().toISOString(),
        });
        setModelReady(true);
        setStage(ASK_BEFORE_REVEAL ? "labelling" : "done");
      } catch (e) {
        console.error(e);
        setError(e instanceof Error ? e.message : String(e));
        setStage("error");
      } finally {
        setStatusText("");
      }
    },
    [thresholds, modelReady]
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
    setError(null);
  };

  return (
    <main className="mx-auto w-full max-w-md px-4 pb-16 pt-6">
      <header className="mb-4">
        <h1 className="text-xl font-semibold tracking-tight">
          BovineInsight — skin check
        </h1>
        <p className="ur text-[16px]" style={{ color: "var(--muted)" }}>
          جانور کی جلد کی جانچ
        </p>
      </header>

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
            Take a clear photo of the animal&apos;s skin, close enough to see
            individual lumps.
          </p>
          <p className="ur text-[15px]" style={{ color: "var(--muted)" }}>
            جانور کی جلد کی صاف تصویر لیں، اتنے قریب سے کہ گلٹیاں نظر آئیں۔
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
              ? "Ready — the photo never leaves your phone until you choose to send it."
              : "Loading the model (about 12 MB, once)…"}
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

      {shot && (stage === "labelling" || stage === "done") && (
        <section className="mt-5 space-y-5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={shot.previewUrl}
            alt="The photo you captured"
            className="w-full rounded-xl"
            style={{ border: "1px solid var(--line)" }}
          />

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
                onSpecies={setSpecies}
                onLumps={setLumps}
              />

              <button
                type="button"
                disabled={!species || !lumps}
                onClick={() => setStage("done")}
                className="tap w-full rounded-xl px-5 text-[17px] font-semibold disabled:opacity-40"
                style={{ background: "var(--accent)", color: "var(--bg)" }}
              >
                Show the result · نتیجہ دکھائیں
              </button>
            </>
          )}

          {stage === "done" && (
            <>
              <ResultCard result={shot.verdict} />

              <LabelChips
                species={species}
                lumps={lumps}
                onSpecies={setSpecies}
                onLumps={setLumps}
              />

              {species && lumps && (
                <SubmitPanel
                  // Remounts when a label changes, so an edited answer cannot
                  // leave a "Sent" confirmation standing over stale values.
                  key={`${species}-${lumps}`}
                  payload={{
                    image: shot.uploadBlob,
                    imageWidth: shot.width,
                    imageHeight: shot.height,
                    capturedAt: shot.capturedAt,
                    probability: shot.probability,
                    latencyMs: shot.latencyMs,
                    quality: shot.quality,
                    verdict: shot.verdict,
                    thresholds: thresholds ?? DEFAULT_THRESHOLDS,
                    species,
                    lumps,
                  }}
                />
              )}

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
  const rows: [string, string][] = [
    ["raw score p(lesion)", shot.probability.toFixed(4)],
    ["threshold tau", (thresholds?.tau ?? 0.5).toFixed(2)],
    [
      "unclear band",
      `[${(thresholds?.unclearLow ?? 0.4).toFixed(2)}, ${(
        thresholds?.unclearHigh ?? 0.6
      ).toFixed(2)})`,
    ],
    ["verdict", shot.verdict.verdict],
    ["abstain reason", shot.verdict.reason ?? "—"],
    ["blur variance", shot.quality.blurVariance.toFixed(1)],
    ["dark / bright frac", `${shot.quality.darkFraction.toFixed(3)} / ${shot.quality.brightFraction.toFixed(3)}`],
    ["mean luma", shot.quality.meanLuma.toFixed(1)],
    ["inference", `${shot.latencyMs.toFixed(0)} ms`],
    ["source image", `${shot.width}×${shot.height}`],
    ["model", MODEL_VERSION],
    ["thresholds", thresholds?.thresholdsVersion ?? "defaults"],
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
