import { INPUT_SIZE } from "./model";
import { resizeBilinearLikePIL } from "./resize";

export interface Prepared {
  /** NHWC, raw [0,255], length 1*224*224*3. Feed this to the model as-is. */
  tensor: Float32Array;
  /** The 224x224 pixels as RGBA, for the quality gate. */
  pixels: ImageData;
}

/**
 * Reading full-resolution pixels costs width*height*4 bytes. A 12 MP photo is
 * ~48 MB of ImageData, which a cheap Android will tolerate; a 50 MP one would
 * not. Above this budget the browser does a mild pre-scale first -- mild
 * enough (under ~1.3x) that its choice of filter barely matters, leaving all
 * the real reduction to our exact one.
 */
const PIXEL_BUDGET = 12_000_000;

function readPixels(source: ImageBitmap, w: number, h: number): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d canvas context unavailable");
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/**
 * Mirrors the Python pipeline behind every number in the model card:
 *
 *     Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)
 *     np.asarray(im, np.float32)          # RAW [0,255]
 *
 * Three things are deliberate and must not be "tidied up":
 *  - the image is SQUASHED to 224x224, not centre-cropped. Aspect ratio was
 *    not preserved in training either.
 *  - the resize is our PIL port, not canvas drawImage. See lib/resize.ts for
 *    the measurement that forced this.
 *  - pixel values stay in [0,255]. The rescaling lives inside the graph.
 */
export async function prepareImage(
  source: ImageBitmap,
  size = INPUT_SIZE
): Promise<Prepared> {
  let src = source;
  let w = source.width;
  let h = source.height;
  let scratch: ImageBitmap | null = null;

  if (w * h > PIXEL_BUDGET) {
    const k = Math.sqrt(PIXEL_BUDGET / (w * h));
    w = Math.max(size, Math.round(w * k));
    h = Math.max(size, Math.round(h * k));
    scratch = await createImageBitmap(source, {
      resizeWidth: w,
      resizeHeight: h,
      resizeQuality: "high",
    });
    src = scratch;
  }

  const full = readPixels(src, w, h);
  scratch?.close();

  const rgb = resizeBilinearLikePIL(full, size, size);

  const tensor = new Float32Array(size * size * 3);
  for (let i = 0; i < tensor.length; i++) {
    tensor[i] = rgb[i]; // raw [0,255] -- NOT divided by 255, no mean/std
  }

  // Rebuild RGBA for the quality gate, which measures the pixels the model saw.
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    rgba[j] = rgb[i];
    rgba[j + 1] = rgb[i + 1];
    rgba[j + 2] = rgb[i + 2];
    rgba[j + 3] = 255;
  }

  return { tensor, pixels: new ImageData(rgba, size, size) };
}

/**
 * A JPEG for upload and later vet review -- NOT the model input.
 *
 * Full-resolution phone photos are 2-5 MB. Uploading one over a 600 kB/s link
 * in a shed is the slowest thing this app would ever do, and the point of
 * collecting them is retraining at 224-256 px anyway. 1024 px on the long edge
 * keeps every detail that matters for both retraining and a vet looking at
 * nodules, for a fraction of the bytes.
 */
export async function makeUploadJpeg(
  source: CanvasImageSource,
  srcWidth: number,
  srcHeight: number,
  maxEdge = 1024,
  quality = 0.85
): Promise<Blob> {
  const scale = Math.min(1, maxEdge / Math.max(srcWidth, srcHeight));
  const w = Math.max(1, Math.round(srcWidth * scale));
  const h = Math.max(1, Math.round(srcHeight * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas context unavailable");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, w, h);

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) =>
        blob ? resolve(blob) : reject(new Error("canvas.toBlob returned null")),
      "image/jpeg",
      quality
    );
  });
}

/** Decode a File/Blob into something drawable, honouring EXIF orientation.
 *  Phone cameras routinely write sideways JPEGs; the model was trained on
 *  upright animals, and the buffalo false-alarm review in MODEL_CARD.md
 *  section 6 found rotated inputs among the worst offenders. */
export async function decodeImage(file: Blob): Promise<ImageBitmap> {
  return createImageBitmap(file, { imageOrientation: "from-image" });
}
