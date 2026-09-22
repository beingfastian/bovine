"""Buffalo specificity test: v1.0.0 vs 324 healthy Pakistani buffalo.

v1.0.0 trained ONLY on cattle. Buffalo are ~59% of Pakistan's milk, black-hided,
and LSD presents differently in them. Buffalo-Pak is a BREED dataset, so the
animals are presumed healthy -- every positive here is a false alarm.

This is the first out-of-distribution test v1.0.0 has faced.
"""
import collections, io, json, zipfile
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL = "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.tflite"
ZIP = "ai/data/buffalo_pak.zip"
TAU = 0.50

it = Interpreter(model_path=MODEL); it.allocate_tensors()
i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
assert tuple(i_d["shape"]) == (1, 224, 224, 3)

z = zipfile.ZipFile(ZIP)
imgs = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]

rows = []
for k, n in enumerate(imgs, 1):
    breed = n.split("/")[1] if "/" in n else "?"
    im = Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((224, 224),
                                                                 Image.BILINEAR)
    x = np.asarray(im, np.float32)[None, ...]      # RAW [0,255]
    it.set_tensor(i_d["index"], x); it.invoke()
    rows.append((n, breed, float(it.get_tensor(o_d["index"])[0][0])))
    if k % 100 == 0:
        print(f"\r  {k}/{len(imgs)}", end="", flush=True)
print()

p = np.array([r[2] for r in rows])
print()
print("=" * 58)
print("BUFFALO SPECIFICITY TEST -- v1.0.0 on healthy Pakistani buffalo")
print("=" * 58)
print(f"n = {len(p)}   (all presumed HEALTHY -- any positive is a FALSE ALARM)")
print()
print(f"  FALSE POSITIVE RATE @ tau={TAU}: {(p >= TAU).mean():.4f}  "
      f"({int((p >= TAU).sum())}/{len(p)})")
print(f"  SPECIFICITY                    : {(p < TAU).mean():.4f}")
print()
print("  probability distribution:")
print(f"    min={p.min():.4f}  p25={np.percentile(p,25):.4f}  "
      f"median={np.median(p):.4f}  p75={np.percentile(p,75):.4f}  max={p.max():.4f}")
print(f"    mean={p.mean():.4f}")
print()
print("  BY BREED:")
for breed in sorted({r[1] for r in rows}):
    q = np.array([r[2] for r in rows if r[1] == breed])
    print(f"    {breed:12} n={len(q):4d}  FPR={(q >= TAU).mean():.4f}  "
          f"median_p={np.median(q):.4f}  max_p={q.max():.4f}")
print()
print("  REFERENCE -- v1.0.0 on CATTLE test set: specificity 0.8551 "
      "(FPR 0.1449)")

# contact sheet of the highest-scoring buffalo -- eyeball whether any are
# genuinely affected, which would make them correct rather than false alarms
rows.sort(key=lambda r: -r[2])
sheet = Image.new("RGB", (6 * 180, 3 * 180), "white")
for i, (n, breed, pr) in enumerate(rows[:18]):
    im = Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((180, 180))
    sheet.paste(im, (180 * (i % 6), 180 * (i // 6)))
sheet.save("ai/reports/buffalo_top_scoring.png")
print()
print("wrote ai/reports/buffalo_top_scoring.png (18 highest-scoring buffalo)")
print("  top scores:", [round(r[2], 3) for r in rows[:8]])

json.dump({"n": len(p), "tau": TAU,
           "false_positive_rate": float((p >= TAU).mean()),
           "specificity": float((p < TAU).mean()),
           "median_p": float(np.median(p)), "mean_p": float(p.mean()),
           "max_p": float(p.max()),
           "by_breed": {b: {"n": int(sum(1 for r in rows if r[1] == b)),
                            "fpr": float(np.mean([r[2] for r in rows if r[1] == b])
                                         >= TAU),
                            "median_p": float(np.median([r[2] for r in rows
                                                         if r[1] == b]))}
                        for b in sorted({r[1] for r in rows})}},
          open("ai/reports/buffalo_specificity.json", "w"), indent=2)
