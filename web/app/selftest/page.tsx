"use client";

import { useCallback, useEffect, useState } from "react";
import { DEFAULT_THRESHOLDS, MODEL_VERSION } from "@/lib/model";
import { decodeImage, prepareImage } from "@/lib/preprocess";
import { runInference } from "@/lib/infer";
import { assessQuality } from "@/lib/quality";
import { decide } from "@/lib/verdict";

/**
 * Browser-vs-Python parity harness.
 *
 * The Kaggle kernel and ai/scripts/verify_onnx.py prove the ONNX graph is
 * numerically the model we measured. Neither proves the BROWSER hands it the
 * same pixels: canvas downscaling is not PIL bilinear, and the JPEG decoders
 * differ. That gap is unmeasurable from Node, so it gets measured here, on the
 * real device, against probabilities computed in Python.
 *
 * Open /selftest on any phone that is going to be used for collection. If the
 * verdicts still agree, the pipeline on that device is sound.
 */

interface Fixture {
  file: string;
  expectedProbability: number;
  label: number;
  source: string;
  severity: string;
}

interface Row extends Fixture {
  actual: number;
  diff: number;
  expectedVerdict: string;
  actualVerdict: string;
  verdictAgrees: boolean;
  /** Sits within BOUNDARY_TOLERANCE of tau or a dead-band edge, where the
   *  verdict is undefined at this precision. */
  onBoundary: boolean;
  latencyMs: number;
}

/**
 * The fixture set deliberately includes the samples closest to tau and to both
 * dead-band edges, because those are where drift would do damage. A sample
 * sitting 0.002 from a boundary will flip under ANY nonzero drift, so counting
 * that as a failure would just be demanding bit-exact equality with Pillow.
 *
 * Flips are therefore only counted where the reference probability is further
 * than this from every boundary. Boundary samples are still shown, and the max
 * drift figure still covers all of them -- it is the number that would catch a
 * genuine preprocessing regression.
 */
const BOUNDARY_TOLERANCE = 0.01;

export default function SelfTest() {
  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    setRows([]);
    try {
      setStatus("loading fixtures…");
      const manifest = await fetch("/fixtures/fixtures.json").then((r) => r.json());
      const fixtures: Fixture[] = manifest.fixtures;
      const t = DEFAULT_THRESHOLDS;
      const out: Row[] = [];

      for (let i = 0; i < fixtures.length; i++) {
        const f = fixtures[i];
        setStatus(`scoring ${i + 1}/${fixtures.length} — ${f.file}`);

        const blob = await fetch(`/fixtures/${f.file}`).then((r) => r.blob());
        const bitmap = await decodeImage(blob);
        const prepared = await prepareImage(bitmap);
        const quality = assessQuality(prepared.pixels, t);
        bitmap.close();

        const { probability, latencyMs } = await runInference(prepared.tensor);

        // Compare verdicts with the quality gate neutralised, so this measures
        // the numeric pipeline rather than the blur of an old web JPEG.
        const passing = { ...quality, ok: true, reason: null };
        const expectedVerdict = decide(f.expectedProbability, passing, t).verdict;
        const actualVerdict = decide(probability, passing, t).verdict;

        const onBoundary = [t.tau, t.unclearLow, t.unclearHigh].some(
          (b) => Math.abs(f.expectedProbability - b) < BOUNDARY_TOLERANCE
        );

        out.push({
          ...f,
          actual: probability,
          diff: Math.abs(probability - f.expectedProbability),
          expectedVerdict,
          actualVerdict,
          verdictAgrees: expectedVerdict === actualVerdict,
          onBoundary,
          latencyMs,
        });
        setRows([...out]);
      }
      setStatus("done");
    } catch (e) {
      console.error(e);
      setError(e instanceof Error ? e.message : String(e));
      setStatus("failed");
    } finally {
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    void run();
  }, [run]);

  const maxDiff = rows.length ? Math.max(...rows.map((r) => r.diff)) : 0;
  const meanDiff = rows.length
    ? rows.reduce((a, r) => a + r.diff, 0) / rows.length
    : 0;
  const disagreements = rows.filter((r) => !r.verdictAgrees && !r.onBoundary).length;
  const boundaryFlips = rows.filter((r) => !r.verdictAgrees && r.onBoundary).length;
  const medianLatency = rows.length
    ? [...rows.map((r) => r.latencyMs)].sort((a, b) => a - b)[
        Math.floor(rows.length / 2)
      ]
    : 0;
  const complete = status === "done" && rows.length > 0;
  const pass = complete && disagreements === 0 && maxDiff < 0.02;

  return (
    <main className="mx-auto w-full max-w-2xl px-4 pb-16 pt-6">
      <h1 className="text-xl font-semibold">Browser parity self-test</h1>
      <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
        Runs {MODEL_VERSION} in this browser over fixtures whose expected
        probabilities were computed in Python, and reports how far the two
        pipelines drift apart.
      </p>

      {complete && (
        <div
          className="mt-4 rounded-xl px-4 py-3"
          style={{
            background: pass ? "var(--concern-low-bg)" : "var(--concern-high-bg)",
            color: pass ? "var(--concern-low)" : "var(--concern-high)",
          }}
        >
          <p className="text-lg font-semibold">{pass ? "PASS" : "FAIL"}</p>
          <p className="mt-1 text-sm">
            {rows.length} fixtures · max drift {maxDiff.toFixed(5)} · mean drift{" "}
            {meanDiff.toFixed(5)} · verdict disagreements {disagreements} ·
            median inference {medianLatency.toFixed(0)} ms
          </p>
          <p className="mt-1 text-xs opacity-80">
            Criterion: max probability drift under 0.02, and no verdict
            disagreement away from a threshold boundary.
            {boundaryFlips > 0 &&
              ` ${boundaryFlips} boundary-sitting fixture${
                boundaryFlips === 1 ? "" : "s"
              } flipped, which drift of this size makes unavoidable.`}
          </p>
        </div>
      )}

      {!complete && (
        <p className="mt-4 text-sm" style={{ color: "var(--muted)" }}>
          {status}
        </p>
      )}

      {error && (
        <div
          className="mt-4 rounded-xl px-4 py-3 text-sm"
          style={{
            background: "var(--concern-high-bg)",
            color: "var(--concern-high)",
          }}
        >
          {error}
        </div>
      )}

      <button
        type="button"
        onClick={() => void run()}
        disabled={running}
        className="tap mt-4 rounded-xl border px-5 text-sm font-medium disabled:opacity-40"
        style={{ borderColor: "var(--line)" }}
      >
        Run again
      </button>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full text-left font-mono text-[12px]">
          <thead style={{ color: "var(--muted)" }}>
            <tr>
              <th className="py-1 pr-3">file</th>
              <th className="py-1 pr-3 text-right">python</th>
              <th className="py-1 pr-3 text-right">browser</th>
              <th className="py-1 pr-3 text-right">diff</th>
              <th className="py-1 pr-3 text-right">ms</th>
              <th className="py-1">verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.file}
                style={{ borderTop: "1px solid var(--line)" }}
              >
                <td className="py-1 pr-3">
                  {r.file.slice(0, 8)}
                  {r.onBoundary && (
                    <span title="sits on a threshold boundary" style={{ color: "var(--muted)" }}>
                      {" "}*
                    </span>
                  )}
                </td>
                <td className="py-1 pr-3 text-right">
                  {r.expectedProbability.toFixed(4)}
                </td>
                <td className="py-1 pr-3 text-right">{r.actual.toFixed(4)}</td>
                <td
                  className="py-1 pr-3 text-right"
                  style={{
                    color: r.diff > 0.02 ? "var(--concern-mid)" : undefined,
                  }}
                >
                  {r.diff.toFixed(5)}
                </td>
                <td className="py-1 pr-3 text-right">
                  {r.latencyMs.toFixed(0)}
                </td>
                <td
                  className="py-1"
                  style={{
                    color: r.verdictAgrees
                      ? "var(--concern-low)"
                      : r.onBoundary
                        ? "var(--concern-mid)"
                        : "var(--concern-high)",
                  }}
                >
                  {r.verdictAgrees
                    ? r.actualVerdict
                    : `${r.expectedVerdict} → ${r.actualVerdict}${r.onBoundary ? " (boundary)" : ""}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}
