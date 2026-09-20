"""Measure whether a smaller ONNX is safe to ship.

The fp32 export is 12.5 MB and barely compresses (float weights are close to
incompressible): ~11.2 MB gzipped, on top of ~3.6 MB for the WASM runtime. On
the 600 kB/s - 1.5 MB/s connections this app targets, that is 10-25 seconds
before the first photo can be scored -- for a link shared over WhatsApp, which
people abandon.

So: build the candidates, and measure each against the SAME locked test split
and the SAME criteria the fp32 model had to pass. Nothing ships on the promise
of being smaller.

  float16   -- halves the weights; ORT's WASM CPU kernels are fp32, so it
               relies on casts and may be slower or unsupported.
  int8 QDQ  -- static, calibrated on TRAINING images (never the test split);
               ~4x smaller, and the variant ORT Web runs natively.

Run:  python ai/scripts/quantize_onnx.py
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.onnx"
OUT_DIR = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0"
BUNDLE = ROOT / "ai/data/bovine_lsd_bundle.zip"
REPORT = ROOT / "ai/reports/quantization.json"
TAU = 0.50
N_CALIB = 160


def preprocess(im: Image.Image) -> np.ndarray:
    return np.asarray(
        im.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32
    )[None, ...]


def load_split(z: zipfile.ZipFile, split: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(z.read("manifest.csv").decode())))
    if split == "test":
        return [r for r in rows if r["split"] == "test"]
    return [r for r in rows if r["split"] != "test"]


def auc(y, p):
    o = np.argsort(p, kind="mergesort")
    r = np.empty(len(p), float)
    s = p[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p, tau=TAU):
    yh = (p >= tau).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"balanced_accuracy": (rec + spec) / 2, "auc": auc(y, p),
            "recall": rec, "specificity": spec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def gzip_size(path: Path) -> int:
    import gzip
    return len(gzip.compress(path.read_bytes(), 9))


def score(path: Path, z: zipfile.ZipFile, rows: list[dict]) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for k, r in enumerate(rows, 1):
        x = preprocess(Image.open(io.BytesIO(z.read(r["file"]))))
        out.append(float(sess.run(None, {name: x})[0].ravel()[0]))
        if k % 100 == 0 or k == len(rows):
            print(f"\r    {k}/{len(rows)}", end="", flush=True)
    print()
    return np.array(out)


def build_fp16(dst: Path) -> bool:
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("  onnxconverter-common not installed; skipping fp16")
        return False
    m = onnx.load(str(SRC))
    # Keep the graph's own input/output in fp32 so the client contract is
    # unchanged: raw [0,255] float32 in, float32 probability out.
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(dst))
    return True


def build_int8(dst: Path, z: zipfile.ZipFile) -> bool:
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader, QuantFormat, QuantType, quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as e:
        print(f"  onnxruntime.quantization unavailable ({e}); skipping int8")
        return False

    train_rows = load_split(z, "train")
    rng = np.random.default_rng(42)
    picks = rng.choice(len(train_rows), size=min(N_CALIB, len(train_rows)), replace=False)
    calib_rows = [train_rows[i] for i in picks]
    print(f"  calibrating on {len(calib_rows)} TRAINING images (test split untouched)")

    class Reader(CalibrationDataReader):
        def __init__(self, name):
            self.name = name
            self.i = 0

        def get_next(self):
            if self.i >= len(calib_rows):
                return None
            r = calib_rows[self.i]
            self.i += 1
            if self.i % 40 == 0:
                print(f"\r    calib {self.i}/{len(calib_rows)}", end="", flush=True)
            return {self.name: preprocess(Image.open(io.BytesIO(z.read(r["file"]))))}

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(SRC), str(pre), skip_symbolic_shape=True)

    name = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(
        model_input=str(pre),
        model_output=str(dst),
        calibration_data_reader=Reader(name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print()
    pre.unlink(missing_ok=True)
    return True


def main() -> None:
    z = zipfile.ZipFile(BUNDLE)
    test_rows = load_split(z, "test")
    y = np.array([int(r["label"]) for r in test_rows])

    print(f"baseline fp32: {SRC.name}")
    p_ref = score(SRC, z, test_rows)
    ref = metrics(y, p_ref)
    print(f"  bal.acc {ref['balanced_accuracy']:.4f}  AUC {ref['auc']:.4f}  "
          f"TP={ref['tp']} TN={ref['tn']} FP={ref['fp']} FN={ref['fn']}")

    candidates = {}
    tmp = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/_quant"
    tmp.mkdir(parents=True, exist_ok=True)

    print("\nbuilding float16…")
    f16 = tmp / "bi-lsd-mnv3l-v1.0.0.fp16.onnx"
    if build_fp16(f16):
        candidates["fp16"] = f16

    print("\nbuilding int8 (static, QDQ, per-channel)…")
    i8 = tmp / "bi-lsd-mnv3l-v1.0.0.int8.onnx"
    try:
        if build_int8(i8, z):
            candidates["int8"] = i8
    except Exception as e:
        print(f"  int8 build failed: {type(e).__name__}: {e}")

    results = {
        "baseline": {
            "file": SRC.name,
            "bytes": SRC.stat().st_size,
            "gzipBytes": gzip_size(SRC),
            "metrics": ref,
        },
        "candidates": {},
    }

    for tag, path in candidates.items():
        print(f"\nscoring {tag}…")
        try:
            p = score(path, z, test_rows)
        except Exception as e:
            print(f"  FAILED to run: {type(e).__name__}: {e}")
            results["candidates"][tag] = {"error": f"{type(e).__name__}: {e}"}
            continue

        m = metrics(y, p)
        agree = float(((p >= TAU) == (p_ref >= TAU)).mean())
        entry = {
            "bytes": path.stat().st_size,
            "gzipBytes": gzip_size(path),
            "metrics": m,
            "decision_agreement_vs_fp32": agree,
            "max_abs_diff_vs_fp32": float(np.max(np.abs(p - p_ref))),
            "mean_abs_diff_vs_fp32": float(np.mean(np.abs(p - p_ref))),
            "bal_acc_delta": m["balanced_accuracy"] - ref["balanced_accuracy"],
        }
        results["candidates"][tag] = entry
        print(f"  {path.stat().st_size/1e6:5.2f} MB ({entry['gzipBytes']/1e6:5.2f} MB gzip)  "
              f"bal.acc {m['balanced_accuracy']:.4f} ({entry['bal_acc_delta']:+.4f})  "
              f"agreement {agree:.4f}  maxdiff {entry['max_abs_diff_vs_fp32']:.4f}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    base_gz = results["baseline"]["gzipBytes"]
    print(f"  fp32 baseline: {base_gz/1e6:.2f} MB gzip, bal.acc {ref['balanced_accuracy']:.4f}")

    # A candidate is only worth shipping if it is meaningfully smaller AND its
    # decisions barely move. Recall is the asymmetric one: a false negative
    # tells a farmer a diseased animal is fine.
    recommendation = "fp32"
    for tag, e in results["candidates"].items():
        if "error" in e:
            print(f"  {tag}: unusable — {e['error']}")
            continue
        saving = 1 - e["gzipBytes"] / base_gz
        ok = (
            e["decision_agreement_vs_fp32"] >= 0.99
            and e["metrics"]["recall"] >= ref["recall"] - 0.01
            and abs(e["bal_acc_delta"]) <= 0.01
            and saving >= 0.30
        )
        print(f"  {tag}: {e['gzipBytes']/1e6:.2f} MB gzip ({saving*100:.0f}% smaller), "
              f"agreement {e['decision_agreement_vs_fp32']:.4f}, "
              f"recall {e['metrics']['recall']:.4f} vs {ref['recall']:.4f} "
              f"-> {'SAFE TO SHIP' if ok else 'reject'}")
        if ok and recommendation == "fp32":
            recommendation = tag

    results["recommendation"] = recommendation
    results["criteria"] = {
        "decision_agreement_vs_fp32": ">= 0.99",
        "recall": ">= fp32 recall - 0.01",
        "balanced_accuracy_delta": "<= 0.01 absolute",
        "size_saving": ">= 30% gzipped",
    }
    print(f"\n  RECOMMENDATION: ship {recommendation}")

    if recommendation != "fp32":
        final = OUT_DIR / f"bi-lsd-mnv3l-v1.0.0.{recommendation}.onnx"
        shutil.copy(candidates[recommendation], final)
        print(f"  copied -> {final.relative_to(ROOT)}")
        results["shippedArtifact"] = final.name

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
