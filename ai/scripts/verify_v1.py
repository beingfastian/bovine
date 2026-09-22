"""Independent verification: run the SHIPPED .tflite over the locked test split.

Kaggle reported these numbers from the Keras model. This re-runs them locally
from the actual artifact that will go into the APK, so what we ship is what we
measured.
"""
import csv, io, json, zipfile
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL = "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.tflite"
BUNDLE = "ai/data/bovine_lsd_bundle.zip"
TAU = 0.50

it = Interpreter(model_path=MODEL); it.allocate_tensors()
i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
print(f"model in={i_d['shape'].tolist()} {i_d['dtype'].__name__}  "
      f"out={o_d['shape'].tolist()}")

z = zipfile.ZipFile(BUNDLE)
rows = [r for r in csv.DictReader(io.StringIO(z.read("manifest.csv").decode()))
        if r["split"] == "test"]
print(f"test images: {len(rows)}")

y, p, sev = [], [], []
for k, r in enumerate(rows, 1):
    im = Image.open(io.BytesIO(z.read(r["file"]))).convert("RGB")
    im = im.resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, np.float32)[None, ...]          # RAW [0,255]
    it.set_tensor(i_d["index"], x); it.invoke()
    p.append(float(it.get_tensor(o_d["index"])[0][0]))
    y.append(int(r["label"])); sev.append(r.get("severity", ""))
    if k % 150 == 0: print(f"\r  {k}/{len(rows)}", end="", flush=True)
print()

y = np.array(y); p = np.array(p); sev = np.array(sev)

def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p), float); s = p[o]
    i = 0
    while i < len(s):
        j = i
        while j+1 < len(s) and s[j+1] == s[i]: j += 1
        r[o[i:j+1]] = (i+j)/2 + 1; i = j+1
    n1, n0 = int((y==1).sum()), int((y==0).sum())
    return (r[y==1].sum() - n1*(n1+1)/2)/(n1*n0)

yh = (p >= TAU).astype(int)
tp = int(((yh==1)&(y==1)).sum()); tn = int(((yh==0)&(y==0)).sum())
fp = int(((yh==1)&(y==0)).sum()); fn = int(((yh==0)&(y==1)).sum())
rec, spec = tp/(tp+fn), tn/(tn+fp)

print("\n" + "="*56)
print("LOCAL VERIFICATION OF THE SHIPPED .tflite")
print("="*56)
print(f"  accuracy           {(tp+tn)/len(y):.4f}")
print(f"  balanced accuracy  {(rec+spec)/2:.4f}")
print(f"  AUC                {auc(y,p):.4f}")
print(f"  recall             {rec:.4f}")
print(f"  specificity        {spec:.4f}")
print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
print()
print("  KAGGLE REPORTED (Keras): acc 0.9264  bal 0.9090  AUC 0.9653")
print("                           TP=389 TN=177 FP=30 FN=15")
print()
for s in ["Mild", "Severe"]:
    m = sev == s
    if m.sum():
        print(f"  {s:7} n={int(m.sum()):3d}  recall={(p[m]>=TAU).mean():.4f}")
print()
print(f"  prob dist: min={p.min():.4f} med={np.median(p):.4f} max={p.max():.4f}")
json.dump({"accuracy": (tp+tn)/len(y), "balanced_accuracy": (rec+spec)/2,
           "auc": float(auc(y,p)), "recall": rec, "specificity": spec,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn, "tau": TAU, "n": len(y)},
          open("ai/reports/verify_v1_tflite.json", "w"), indent=2)
print("\nwrote ai/reports/verify_v1_tflite.json")
