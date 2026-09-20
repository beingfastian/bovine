import type * as Ort from "onnxruntime-web/wasm";
import { INPUT_SIZE, MODEL_URL, MODEL_VERSION } from "./model";
import { ORT_VERSION } from "./ort-version";

/**
 * onnxruntime-web session management.
 *
 * Plain WASM only -- no WebGPU, no threads. Threads would need COOP/COEP
 * cross-origin isolation headers, which complicate hosting and break embedded
 * content, to speed up a model that already runs in well under a second on one
 * core. The jsep/WebGPU build is also 28 MB of extra download for users on a
 * 600 kB/s connection.
 *
 * onnxruntime-web touches `document` and `navigator` at import time, and
 * `output: "export"` prerenders every page in Node at build time. So the
 * module is imported lazily, inside the browser, on first use.
 */

let ortPromise: Promise<typeof Ort> | null = null;

function getOrt(): Promise<typeof Ort> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/wasm").then((ort) => {
      ort.env.wasm.wasmPaths = `/ort/${ORT_VERSION}/`;
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = "error";
      return ort;
    });
  }
  return ortPromise;
}

let sessionPromise: Promise<Ort.InferenceSession> | null = null;

export function loadSession(): Promise<Ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = getOrt()
      .then((ort) =>
        ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        })
      )
      .catch((err) => {
        // Let the next attempt retry rather than caching a failure forever --
        // the usual cause is a dropped download, not a broken model.
        sessionPromise = null;
        throw err;
      });
  }
  return sessionPromise;
}

export interface InferenceResult {
  /** P(lesion). The model's final op is a sigmoid, so this is ALREADY a
   *  probability. Applying another sigmoid would squash it into [0.5, 0.73]
   *  and silently invalidate tau. */
  probability: number;
  /** Wall-clock milliseconds for the forward pass alone. */
  latencyMs: number;
  modelVersion: string;
}

export async function runInference(
  tensorData: Float32Array
): Promise<InferenceResult> {
  const expected = INPUT_SIZE * INPUT_SIZE * 3;
  if (tensorData.length !== expected) {
    throw new Error(
      `expected ${expected} values (1x${INPUT_SIZE}x${INPUT_SIZE}x3 NHWC), got ${tensorData.length}`
    );
  }

  const ort = await getOrt();
  const session = await loadSession();

  const input = new ort.Tensor("float32", tensorData, [
    1,
    INPUT_SIZE,
    INPUT_SIZE,
    3,
  ]);

  // Read the names off the graph rather than hardcoding them, so a re-export
  // that renames the signature cannot silently break inference.
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const t0 = performance.now();
  const output = await session.run({ [inputName]: input });
  const latencyMs = performance.now() - t0;

  const raw = output[outputName].data as Float32Array;
  const probability = raw[0];

  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error(
      `model returned ${probability}, which is not a probability -- the graph or the input contract is wrong`
    );
  }

  return { probability, latencyMs, modelVersion: MODEL_VERSION };
}

/** Warm the session (downloads and compiles the graph) without scoring
 *  anything, so the first real photo is not also the first download. */
export async function warmUp(): Promise<void> {
  await loadSession();
}
