"""A-T4: Measure the existing model's real accuracy on the Mendeley dataset.

Uses the VERIFIED contract from ai/reports/phase-a.md:
  input  : [1,224,224,3] float32, RAW 0-255 (model rescales internally: x/127.5 - 1)
  output : [1,1] float32, ALREADY a sigmoid probability = P(class 1)

No sklearn dependency - metrics computed directly with numpy.
"""
import io, json, os, sys, zipfile
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL = "bovineinsight/assets/finals_models00.tflite"
ZIP   = "ai/data/lumpy_skin_dataset.zip"
OUT   = "ai/reports/at4_results.json"

# ---------------------------------------------------------------- load images
def list_images(zf):
    """Return [(name, label)] where label 1 = lumpy/diseased, 0 = normal."""
    items = []
    for n in zf.namelist():
        if n.endswith("/") or "__MACOSX" in n:
            continue
        if not n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        low = n.lower()
        if "lumpy" in low:
            lab = 1
        elif "normal" in low or "healthy" in low:
            lab = 0
        else:
            continue          # unclassifiable path - skip and report
        items.append((n, lab))
    return items


def preprocess(pil):
    """RAW 0-255 RGB, 224x224, NHWC. NO normalisation - model rescales internally."""
    img = pil.convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None, ...]


# ---------------------------------------------------------------- metrics
def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    acc  = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal  = (rec + spec) / 2
    return dict(threshold=thr, accuracy=acc, balanced_accuracy=bal, precision=prec,
                recall=rec, specificity=spec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def roc_auc(y, p):
    """Mann-Whitney U / rank-based AUC. Handles ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sp = p[order]
    i = 0
    while i < len(sp):                       # average ranks within tie groups
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- main
def main():
    if not os.path.exists(ZIP):
        sys.exit(f"dataset not found: {ZIP} - run ai/scripts/fetch_dataset.py first")

    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    i_d = interp.get_input_details()[0]
    o_d = interp.get_output_details()[0]
    assert tuple(i_d["shape"]) == (1, 224, 224, 3), i_d["shape"]
    assert tuple(o_d["shape"]) == (1, 1), o_d["shape"]

    with zipfile.ZipFile(ZIP) as zf:
        items = list_images(zf)
        print(f"found {len(items)} labelled images "
              f"({sum(l for _, l in items)} lumpy / {sum(1 - l for _, l in items)} normal)")

        y, p, failed = [], [], 0
        for k, (name, lab) in enumerate(items, 1):
            try:
                with zf.open(name) as fh:
                    x = preprocess(Image.open(io.BytesIO(fh.read())))
            except Exception:
                failed += 1
                continue
            interp.set_tensor(i_d["index"], x)
            interp.invoke()
            p.append(float(interp.get_tensor(o_d["index"])[0][0]))
            y.append(lab)
            if k % 100 == 0:
                print(f"\r  {k}/{len(items)}", end="", flush=True)
        print()

    y = np.array(y); p = np.array(p)
    print(f"scored {len(y)} images ({failed} unreadable)\n")

    print("PROBABILITY DISTRIBUTION")
    print(f"  overall  min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f}")
    print(f"  lumpy    mean={p[y==1].mean():.4f}  n={int((y==1).sum())}")
    print(f"  normal   mean={p[y==0].mean():.4f}  n={int((y==0).sum())}")
    print(f"  exactly 0.0: {int((p==0).sum())}   exactly 1.0: {int((p==1).sum())}\n")

    auc = roc_auc(y, p)
    auc_inv = roc_auc(y, 1 - p)
    print(f"ROC-AUC (as-is)   = {auc:.4f}")
    print(f"ROC-AUC (inverted)= {auc_inv:.4f}")
    if auc_inv > auc + 0.05:
        print("  >> POLARITY LIKELY FLIPPED: output is P(Normal), not P(Lumpy)")
    print()

    best = max((metrics(y, p, t) for t in np.arange(0.05, 1.0, 0.01)),
               key=lambda m: m["balanced_accuracy"])
    at50 = metrics(y, p, 0.5)

    for tag, m in (("@0.50 (default)", at50), (f"@{best['threshold']:.2f} (best bal-acc)", best)):
        print(f"{tag}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} spec={m['specificity']:.4f} f1={m['f1']:.4f}")
        print(f"    TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

    res = dict(n=len(y), n_lumpy=int((y==1).sum()), n_normal=int((y==0).sum()),
               unreadable=failed, roc_auc=auc, roc_auc_inverted=auc_inv,
               at_0_5=at50, best=best,
               prob_stats=dict(min=float(p.min()), max=float(p.max()),
                               mean_lumpy=float(p[y==1].mean()),
                               mean_normal=float(p[y==0].mean()),
                               n_exactly_zero=int((p==0).sum()),
                               n_exactly_one=int((p==1).sum())))
    os.makedirs("ai/reports", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

    print("\n--- A-T4 DECISION GATE (plan section 2.4) ---")
    score = max(best["balanced_accuracy"], max(auc, auc_inv))
    print(f"best balanced accuracy = {best['balanced_accuracy']:.4f}")
    print("VERDICT:", "weights USABLE as V0 baseline (>=0.80)" if score >= 0.80
          else "weights NOT usable - Phase B is a full retrain (<0.80)")


if __name__ == "__main__":
    main()
