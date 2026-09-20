"""Independent local verification of the ONNX artifact that ships to the browser.

The Kaggle kernel already compared Keras / ONNX / TFLite. This re-runs the ONNX
half locally, against the file actually sitting on disk, for two reasons:

  1. it proves the downloaded bytes are the bytes that were verified, and
  2. it is the script that will be re-run on every future re-export.

It also measures the two things the web UI needs and the model card does not
provide:

  - the "unclear" dead-band around tau: how much abstention buys how much
    precision on the calls we still make.
  - the blur (variance-of-Laplacian) distribution, so the quality gate's
    threshold is a percentile of real data rather than a guess.

Run:  python ai/scripts/verify_onnx.py
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "ai/kernels/bovine-lsd-onnx-export/out/bi-lsd-mnv3l-v1.0.0.onnx"
TFLITE_PATH = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.tflite"
BUNDLE = ROOT / "ai/data/bovine_lsd_bundle.zip"
REPORT = ROOT / "ai/reports/onnx_verification.json"
TAU = 0.50


# ----------------------------------------------------------------- graph checks
def check_graph(path: Path) -> dict:
    m = onnx.load(str(path))
    onnx.checker.check_model(m)

    ops = Counter(n.op_type for n in m.graph.node)
    banned = {k: v for k, v in ops.items()
              if "Random" in k or "Stateless" in k or k in ("If", "Loop")}
    assert not banned, f"training-time / control-flow ops in the graph: {banned}"

    def dims(vi):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in vi.type.tensor_type.shape.dim]

    inp, out = m.graph.input[0], m.graph.output[0]
    info = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sizeBytes": path.stat().st_size,
        "irVersion": m.ir_version,
        "opsets": [(o.domain or "ai.onnx", o.version) for o in m.opset_import],
        "inputName": inp.name,
        "inputShape": dims(inp),
        "outputName": out.name,
        "outputShape": dims(out),
        "nodeCount": sum(ops.values()),
        "opHistogram": dict(ops),
    }

    assert dims(inp) == [1, 224, 224, 3], f"unexpected input shape {dims(inp)}"
    assert dims(out) == [1, 1], f"unexpected output shape {dims(out)}"
    return info


# ------------------------------------------------------------------- image prep
def preprocess(im: Image.Image) -> np.ndarray:
    """The one true preprocessing. Mirrored in web/lib/preprocess.ts."""
    im = im.convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32)[None, ...]  # RAW [0,255]


def laplacian_variance(arr: np.ndarray) -> float:
    """Byte-for-byte the same measure as web/lib/quality.ts: Rec.601 luma,
    4-neighbour Laplacian, variance over interior pixels."""
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    lap = (4 * luma[1:-1, 1:-1]
           - luma[1:-1, :-2] - luma[1:-1, 2:]
           - luma[:-2, 1:-1] - luma[2:, 1:-1])
    return float(lap.var())


# ----------------------------------------------------------------------- metrics
def auc(y: np.ndarray, p: np.ndarray) -> float:
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


def metrics(y: np.ndarray, p: np.ndarray, tau: float = TAU) -> dict:
    yh = (p >= tau).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum())
    tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (rec + spec) / 2,
            "auc": auc(y, p), "recall": rec, "specificity": spec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    print("=" * 66)
    print("ONNX GRAPH")
    print("=" * 66)
    graph = check_graph(ONNX_PATH)
    for k in ("path", "sizeBytes", "opsets", "inputName", "inputShape",
              "outputName", "outputShape", "nodeCount"):
        print(f"  {k:12} {graph[k]}")
    print("  no random / control-flow ops: OK")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    interp = Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]

    z = zipfile.ZipFile(BUNDLE)
    rows = [r for r in csv.DictReader(io.StringIO(z.read("manifest.csv").decode()))
            if r["split"] == "test"]

    print(f"\nscoring {len(rows)} locked test images (ONNX + TFLite)...")
    p_onnx, p_tfl, y, sev, src, blur = [], [], [], [], [], []
    for k, r in enumerate(rows, 1):
        arr = preprocess(Image.open(io.BytesIO(z.read(r["file"]))))
        p_onnx.append(float(sess.run(None, {in_name: arr})[0].ravel()[0]))
        interp.set_tensor(i_d["index"], arr)
        interp.invoke()
        p_tfl.append(float(interp.get_tensor(o_d["index"]).ravel()[0]))
        blur.append(laplacian_variance(arr[0]))
        y.append(int(r["label"]))
        sev.append(r.get("severity", ""))
        src.append(r.get("source", ""))
        if k % 100 == 0 or k == len(rows):
            print(f"\r  {k}/{len(rows)}", end="", flush=True)
    print()

    p_onnx = np.array(p_onnx)
    p_tfl = np.array(p_tfl)
    y = np.array(y)
    sev = np.array(sev)
    blur = np.array(blur)

    m_onnx = metrics(y, p_onnx)
    m_tfl = metrics(y, p_tfl)
    agreement = float(((p_onnx >= TAU) == (p_tfl >= TAU)).mean())

    print("\n" + "=" * 66)
    print("PARITY: ONNX (ships to browser) vs TFLite (shipped to Android)")
    print("=" * 66)
    print(f"  max abs diff        {np.max(np.abs(p_onnx - p_tfl)):.6f}")
    print(f"  mean abs diff       {np.mean(np.abs(p_onnx - p_tfl)):.6f}")
    print(f"  decision agreement  {agreement:.4f}")
    print(f"\n  ONNX   bal.acc {m_onnx['balanced_accuracy']:.4f}  AUC {m_onnx['auc']:.4f}  "
          f"TP={m_onnx['tp']} TN={m_onnx['tn']} FP={m_onnx['fp']} FN={m_onnx['fn']}")
    print(f"  TFLite bal.acc {m_tfl['balanced_accuracy']:.4f}  AUC {m_tfl['auc']:.4f}  "
          f"TP={m_tfl['tp']} TN={m_tfl['tn']} FP={m_tfl['fp']} FN={m_tfl['fn']}")
    print("  model card         bal.acc 0.9090  AUC 0.9653  TP=389 TN=177 FP=30 FN=15")

    # ------------------------------------------------------------ dead-band sweep
    # A symmetric band around tau routed to "unclear". What does the abstention
    # buy on the calls we still make?
    print("\n" + "=" * 66)
    print("UNCLEAR DEAD-BAND SWEEP (ONNX probabilities)")
    print("=" * 66)
    print(f"  {'band':>16}  {'abstain':>8}  {'recall':>7}  {'spec':>7}  {'bal.acc':>7}  "
          f"{'missed+':>8}")
    sweep = []
    for half in [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
        lo, hi = TAU - half, TAU + half
        undecided = (p_onnx >= lo) & (p_onnx < hi)
        decided = ~undecided
        abstain = float(undecided.mean())
        if decided.sum() == 0:
            continue
        md = metrics(y[decided], p_onnx[decided])
        # Positives pushed into "unclear" -- they still get "consult a vet",
        # so this is a softened call, not a missed one. Tracked anyway.
        softened_pos = int((undecided & (y == 1)).sum())
        row = {"half_width": half, "low": lo, "high": hi,
               "abstention_rate": abstain,
               "recall_decided": md["recall"], "specificity_decided": md["specificity"],
               "balanced_accuracy_decided": md["balanced_accuracy"],
               "positives_routed_to_unclear": softened_pos,
               "n_decided": int(decided.sum())}
        sweep.append(row)
        print(f"  [{lo:.3f},{hi:.3f})  {abstain:8.3f}  {md['recall']:7.4f}  "
              f"{md['specificity']:7.4f}  {md['balanced_accuracy']:7.4f}  {softened_pos:8d}")

    # ---------------------------------------------------------------- blur stats
    print("\n" + "=" * 66)
    print("BLUR: variance of Laplacian over the 611 test images")
    print("=" * 66)
    pct = {f"p{q}": float(np.percentile(blur, q)) for q in (0.5, 1, 2, 5, 10, 25, 50, 90)}
    print(f"  min {blur.min():.1f}   median {np.median(blur):.1f}   max {blur.max():.1f}")
    for k, v in pct.items():
        print(f"  {k:>5} {v:10.1f}")

    # ------------------------------------------------------------ per-severity
    per_sev = {}
    for s in ("Mild", "Severe"):
        m = sev == s
        if m.sum():
            per_sev[s] = {"n": int(m.sum()), "recall": float((p_onnx[m] >= TAU).mean())}
    print("\n  per severity (ONNX):", per_sev)

    report = {
        "modelVersion": "bi-lsd-mnv3l-v1.0.0",
        "graph": graph,
        "n": int(len(y)),
        "tau": TAU,
        "parity_vs_tflite": {
            "max_abs_diff": float(np.max(np.abs(p_onnx - p_tfl))),
            "mean_abs_diff": float(np.mean(np.abs(p_onnx - p_tfl))),
            "decision_agreement": agreement,
        },
        "metrics_onnx": m_onnx,
        "metrics_tflite": m_tfl,
        "per_severity_onnx": per_sev,
        "deadband_sweep": sweep,
        "blur_variance": {"min": float(blur.min()), "max": float(blur.max()),
                          "median": float(np.median(blur)), **pct},
        "runtime": {"onnxruntime": ort.__version__, "onnx": onnx.__version__},
    }

    checks = [
        ("decision agreement with TFLite == 1.0", agreement == 1.0),
        ("confusion matrix identical to TFLite",
         all(m_onnx[k] == m_tfl[k] for k in ("tp", "tn", "fp", "fn"))),
        ("balanced accuracy within 0.005 of the model card",
         abs(m_onnx["balanced_accuracy"] - 0.9090) < 0.005),
        ("input is [1,224,224,3] NHWC float32", graph["inputShape"] == [1, 224, 224, 3]),
        ("output is [1,1]", graph["outputShape"] == [1, 1]),
        ("probabilities stay inside [0,1]",
         bool(p_onnx.min() >= 0.0 and p_onnx.max() <= 1.0)),
    ]
    print("\n" + "=" * 66)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    report["allChecksPassed"] = all(c[1] for c in checks)
    print("\nOVERALL:", "PASS" if report["allChecksPassed"] else "FAIL")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
