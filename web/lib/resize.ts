/**
 * A port of Pillow's `Image.resize(..., Image.BILINEAR)`.
 *
 * Why this file exists, in one measured number: with canvas `drawImage()` doing
 * the downscale, browser probabilities drifted from the Python reference by up
 * to 0.027, and one fixture crossed a verdict boundary. Those test images are
 * 256 px, a 1.14x reduction. A real phone photo is a ~13x reduction, where the
 * filters diverge considerably more.
 *
 * Canvas resampling is deliberately unspecified -- it varies by browser, by
 * platform, and by whether the reduction is large enough to trigger a
 * multi-step path. Every model number we have was produced by PIL bilinear, so
 * the browser doing "something reasonable instead" is precisely the class of
 * silent preprocessing mismatch that destroyed the predecessor model.
 *
 * This is Pillow's algorithm (Resample.c: precompute_coeffs + horizontal pass +
 * vertical pass, with the intermediate 8-bit round it performs for uint8
 * images). Not bit-exact -- Pillow uses fixed-point integer coefficients -- but
 * the same filter with the same support scaling, which is what the drift was
 * actually about.
 */

/** Pillow's bilinear kernel. support = 1.0 */
function triangle(x: number): number {
  const a = x < 0 ? -x : x;
  return a < 1.0 ? 1.0 - a : 0.0;
}

interface Coeffs {
  bounds: Int32Array; // [min, count] per output pixel
  weights: Float64Array; // ksize per output pixel
  ksize: number;
}

/**
 * Pillow scales the filter's support by the reduction ratio, which is what
 * makes its downscale antialiased. A plain bilinear sample of 4 neighbours --
 * what a naive implementation does -- drops most of the source pixels on a 13x
 * reduction and aliases badly.
 */
function precomputeCoeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterScale = Math.max(1.0, scale);
  const support = 1.0 * filterScale;
  const ksize = Math.ceil(support) * 2 + 1;

  const bounds = new Int32Array(outSize * 2);
  const weights = new Float64Array(outSize * ksize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;

    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;

    const base = xx * ksize;
    let sum = 0;
    for (let x = 0; x < xmax; x++) {
      const w = triangle((x + xmin - center + 0.5) / filterScale);
      weights[base + x] = w;
      sum += w;
    }
    if (sum !== 0) {
      for (let x = 0; x < xmax; x++) weights[base + x] /= sum;
    }

    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }

  return { bounds, weights, ksize };
}

/** Pillow's clip8: round to nearest, clamp to [0,255]. */
function clip8(v: number): number {
  const r = v + 0.5;
  if (r <= 0) return 0;
  if (r >= 255.5) return 255;
  return r | 0;
}

/**
 * Resize RGB(A) pixels to dstW x dstH the way Pillow does.
 *
 * Returns an interleaved RGB Uint8ClampedArray of length dstW*dstH*3.
 * Alpha is dropped, matching `.convert("RGB")` on an opaque JPEG.
 */
export function resizeBilinearLikePIL(
  src: ImageData,
  dstW: number,
  dstH: number
): Uint8ClampedArray {
  const srcW = src.width;
  const srcH = src.height;
  const srcData = src.data;

  // --- horizontal pass: srcW x srcH -> dstW x srcH, rounded back to 8 bits,
  // because that is what Pillow does between passes for uint8 images.
  const hx = precomputeCoeffs(srcW, dstW);
  const mid = new Uint8ClampedArray(dstW * srcH * 3);

  for (let y = 0; y < srcH; y++) {
    const srcRow = y * srcW * 4;
    const midRow = y * dstW * 3;
    for (let xx = 0; xx < dstW; xx++) {
      const xmin = hx.bounds[xx * 2];
      const xcount = hx.bounds[xx * 2 + 1];
      const kbase = xx * hx.ksize;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < xcount; k++) {
        const w = hx.weights[kbase + k];
        const p = srcRow + (xmin + k) * 4;
        r += srcData[p] * w;
        g += srcData[p + 1] * w;
        b += srcData[p + 2] * w;
      }
      const o = midRow + xx * 3;
      mid[o] = clip8(r);
      mid[o + 1] = clip8(g);
      mid[o + 2] = clip8(b);
    }
  }

  // --- vertical pass: dstW x srcH -> dstW x dstH
  const vy = precomputeCoeffs(srcH, dstH);
  const out = new Uint8ClampedArray(dstW * dstH * 3);

  for (let yy = 0; yy < dstH; yy++) {
    const ymin = vy.bounds[yy * 2];
    const ycount = vy.bounds[yy * 2 + 1];
    const kbase = yy * vy.ksize;
    const outRow = yy * dstW * 3;
    for (let x = 0; x < dstW; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let k = 0; k < ycount; k++) {
        const w = vy.weights[kbase + k];
        const p = (ymin + k) * dstW * 3 + x * 3;
        r += mid[p] * w;
        g += mid[p + 1] * w;
        b += mid[p + 2] * w;
      }
      const o = outRow + x * 3;
      out[o] = clip8(r);
      out[o + 1] = clip8(g);
      out[o + 2] = clip8(b);
    }
  }

  return out;
}
