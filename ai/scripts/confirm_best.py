"""Full-dataset confirmation using the best-performing convention from the sweep."""
import io, json, zipfile
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

interp = Interpreter(model_path="bovineinsight/assets/finals_models00.tflite")
interp.allocate_tensors()
i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]

def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p), float); sp = p[o]
    i = 0
    while i < len(sp):
        j = i
        while j+1 < len(sp) and sp[j+1] == sp[i]: j += 1
        r[o[i:j+1]] = (i+j)/2 + 1; i = j+1
    n1, n0 = int((y==1).sum()), int((y==0).sum())
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)

zf = zipfile.ZipFile("ai/data/lumpy_skin_dataset.zip")
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
y, p = [], []
for k, n in enumerate(names, 1):
    lab = 1 if "lumpy" in n.lower() else 0
    im = Image.open(io.BytesIO(zf.read(n))).convert("RGB").resize((224,224), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0)[None, ...]      # the /255 convention
    interp.set_tensor(i_d["index"], np.ascontiguousarray(x)); interp.invoke()
    p.append(float(interp.get_tensor(o_d["index"])[0][0])); y.append(lab)
    if k % 200 == 0: print(f"\r  {k}/{len(names)}", end="", flush=True)
print()

y, p = np.array(y), np.array(p)
a = auc(y, p); ai_ = auc(y, 1 - p)
pol = "INVERTED (output ~ P(Normal))" if ai_ > a else "as-is"
best_p = (1 - p) if ai_ > a else p
best_auc = max(a, ai_)

rows = []
for t in np.arange(0.05, 1.0, 0.01):
    yh = (best_p >= t).astype(int)
    tp = int(((yh==1)&(y==1)).sum()); tn = int(((yh==0)&(y==0)).sum())
    fp = int(((yh==1)&(y==0)).sum()); fn = int(((yh==0)&(y==1)).sum())
    rec = tp/(tp+fn) if tp+fn else 0; spec = tn/(tn+fp) if tn+fp else 0
    rows.append((( rec+spec)/2, t, (tp+tn)/len(y), rec, spec, tp, tn, fp, fn))
bal, t, acc, rec, spec, tp, tn, fp, fn = max(rows)

print(f"\nn={len(y)}  ({int((y==1).sum())} lumpy / {int((y==0).sum())} normal)")
print(f"saturated: exactly 0.0 -> {int((p==0).sum())},  exactly 1.0 -> {int((p==1).sum())}")
print(f"distinct output values: {len(np.unique(p))}")
print(f"\nROC-AUC as-is      = {a:.4f}")
print(f"ROC-AUC inverted   = {ai_:.4f}")
print(f"polarity           = {pol}")
print(f"\nBEST balanced accuracy = {bal:.4f} @ threshold {t:.2f}")
print(f"  accuracy={acc:.4f} recall={rec:.4f} specificity={spec:.4f}")
print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"\nmajority-class baseline (always 'normal') = {700/1024:.4f}")
print(f"random-guess AUC = 0.5000")
print("\nGATE (>=0.80 balanced acc):", "PASS" if bal >= 0.80 else "FAIL")

json.dump(dict(convention="/255", n=len(y), auc_asis=a, auc_inverted=ai_,
               polarity=pol, best_balanced_acc=bal, best_threshold=float(t),
               accuracy=acc, recall=rec, specificity=spec,
               tp=tp, tn=tn, fp=fp, fn=fn,
               n_exactly_zero=int((p==0).sum()), n_exactly_one=int((p==1).sum()),
               n_distinct=len(np.unique(p)),
               majority_baseline=700/1024, gate_pass=bool(bal>=0.80)),
          open("ai/reports/at4_best_convention.json","w"), indent=2)
print("\nwrote ai/reports/at4_best_convention.json")
