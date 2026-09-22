import type * as Ort from "onnxruntime-web/wasm";
import {
  INPUT_SIZE,
  MODEL_URL,
  MODEL_VERSION,
  OUTPUT_NAMES,
  SPECIES_CLASSES,
  type SpeciesProbs,
} from "./model";
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
      .then((session) => {
        // Fail at load, loudly, rather than at the first photo, quietly.
        const names = new Set(session.outputNames);
        for (const want of Object.values(OUTPUT_NAMES)) {
          if (!names.has(want)) {
            throw new Error(
              `model is missing output "${want}" (has ${[...names].join(", ")}) -- wrong artifact, or an export that did not rename its outputs`
            );
          }
        }
        return session;
      })
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
  /** P(lesion). The model's final op on this path is a sigmoid, so this is
   *  ALREADY a probability. Applying another sigmoid would squash it into
   *  [0.5, 0.73] and silently invalidate tau. */
  probability: number;
  /** Softmax over cattle / buffalo / other. Sums to 1. */
  species: SpeciesProbs;
  /** Mahalanobis d^2 from the bovine training distribution. Larger = less
   *  like any animal photo the model was trained on. Unbounded above. */
  ood: number;
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

  const t0 = performance.now();
  const output = await session.run({ [session.inputNames[0]]: input });
  const latencyMs = performance.now() - t0;

  const probability = (output[OUTPUT_NAMES.lesion].data as Float32Array)[0];
  const sp = output[OUTPUT_NAMES.species].data as Float32Array;
  const ood = (output[OUTPUT_NAMES.ood].data as Float32Array)[0];

  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error(
      `lesion output ${probability} is not a probability -- the graph or the input contract is wrong`
    );
  }
  const sum = sp[0] + sp[1] + sp[2];
  if (sp.length !== 3 || !Number.isFinite(sum) || Math.abs(sum - 1) > 1e-3) {
    throw new Error(
      `species output ${Array.from(sp)} is not a 3-way softmax -- the graph or the input contract is wrong`
    );
  }
  if (!Number.isFinite(ood) || ood < 0) {
    throw new Error(`ood output ${ood} is not a squared distance`);
  }

  const species = {
    [SPECIES_CLASSES[0]]: sp[0],
    [SPECIES_CLASSES[1]]: sp[1],
    [SPECIES_CLASSES[2]]: sp[2],
  } as SpeciesProbs;

  return { probability, species, ood, latencyMs, modelVersion: MODEL_VERSION };
}

/** Warm the session (downloads and compiles the graph) without scoring
 *  anything, so the first real photo is not also the first download. */
export async function warmUp(): Promise<void> {
  await loadSession();
}
