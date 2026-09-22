"""Generate the v1.1 lesion-crop training notebook + kernel metadata.

    python ai/scripts/build_notebook_v11.py
    -> ai/kernels/bovine-lsd-v1-1-crops/bovine_lsd_v1_1.ipynb
    -> ai/kernels/bovine-lsd-v1-1-crops/kernel-metadata.json

Runs headless on Kaggle, so every figure is saved to /kaggle/working rather
than shown.

DELIBERATELY DOES NOT TOUCH THE TEST SET. v1.1 is an experiment: if its CV
numbers beat v1.0.0 we promote it and evaluate test once. Evaluating test now
and then tuning would burn the only clean measurement we have.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "ai" / "kernels" / "bovine-lsd-v1-1-crops"
NB = OUTDIR / "bovine_lsd_v1_1.ipynb"

CELLS = []
Q3 = '"' * 3


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


md("""
# BovineInsight v1.1 — Lesion-Crop Training

**Hypothesis.** v1.0.0 Grad-CAM showed broad whole-body activation rather than
focal attention on nodules. That predicted and explained its severity gap
(Severe recall 0.99 vs Mild 0.75): it reads *"this animal looks affected"*
rather than *"these are discrete nodules"*.

Training on **lesion crops** should force focal learning.

**The question this notebook answers:** does focal training improve recall on
**small lesions**? Nothing else.

| | v1.0.0 | v1.1 |
|---|---|---|
| Unit | whole animal | lesion box + 25% context |
| Positives | 1,833 groups | 3,606 crops |
| Balance | 0.53 : 1 | 1.03 : 1 |

**This notebook does NOT evaluate the locked test set.** v1.1 is an experiment.
If CV beats v1.0.0 we promote it and measure test once, cleanly.
""")

code("""
# ============ SETUP ============
import os, random, json, math, io, zipfile, glob
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless kernel: save figures, never show
import matplotlib.pyplot as plt

print("TF", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
assert gpus, "no GPU attached"
""")

code("""
# ============ LOAD CROP BUNDLE ============
hits = glob.glob("/kaggle/input/**/manifest.csv", recursive=True)
if hits:
    WORK = os.path.dirname(hits[0])
else:
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    assert zips, "no crop bundle found under /kaggle/input"
    WORK = "/kaggle/working/data"
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
print("data at", WORK)

df = pd.read_csv(f"{WORK}/manifest.csv")
df["path"] = WORK + "/" + df["file"]
df["label"] = df["label"].astype(int)
assert os.path.exists(df["path"].iloc[0]), df["path"].iloc[0]
print("crops:", df.shape)

print()
print(df.groupby("split")["label"].agg(n="size", lesion="sum", p=("mean")).round(3))
print()
print(df.groupby("kind")["label"].agg(n="size").to_dict())

assert df.groupby("group_id")["split"].nunique().max() == 1, "GROUP LEAK"
print()
print("[PASS] no group spans multiple splits")

# SMALL-LESION band: bottom quartile of lesion box area. This is the v1.1 target.
pos = df[df.label == 1]
Q1 = pos["box_area"].quantile(0.25)
df["small"] = (df.label == 1) & (df.box_area <= Q1)
print(f"small-lesion cutoff (p25 area) = {Q1:.4f}")
print("small lesions per split:", df[df.small].groupby("split").size().to_dict())
""")

code("""
# ============ PIPELINE ============
# Raw [0,255] -- MobileNetV3 include_preprocessing=True rescales internally.
IMG, BATCH = 224, 32
AUTO = tf.data.AUTOTUNE

def _load(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG, IMG))
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)

# NOTE: much lighter zoom than v1.0.0. These crops are already framed on the
# lesion; an aggressive RandomZoom would crop the pathology out of the image.
augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.08),
    layers.RandomBrightness(0.3, value_range=(0, 255)),
    layers.RandomContrast(0.3),
], name="augment")

def _gamma(img, label):
    g = tf.random.uniform([], 0.5, 1.8)      # dark-hide robustness
    return tf.clip_by_value(255.0 * ((img / 255.0) ** g), 0.0, 255.0), label

def make_ds(sub, training):
    ds = tf.data.Dataset.from_tensor_slices((sub["path"].values,
                                             sub["label"].values.astype("float32")))
    if training:
        ds = ds.shuffle(len(sub), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTO)
    if training:
        ds = ds.map(_gamma, num_parallel_calls=AUTO)
    return ds.batch(BATCH).prefetch(AUTO)
""")

code("""
# ============ THE GUARD ============
def assert_input_range(ds, name):
    xb, _ = next(iter(ds))
    lo, hi = float(tf.reduce_min(xb)), float(tf.reduce_max(xb))
    assert hi > 1.5, f"INPUT RANGE BUG in {name}: max={hi:.4f} -- expects RAW [0,255]"
    print(f"[PASS] {name} range [{lo:.1f}, {hi:.1f}]")

assert_input_range(make_ds(df[df.split == "fold0"], True), "probe")
""")

code("""
# ============ MODEL (identical contract to v1.0.0) ============
def build_model():
    base = keras.applications.MobileNetV3Large(
        input_shape=(IMG, IMG, 3), include_top=False,
        weights="imagenet", include_preprocessing=True)
    base.trainable = False
    inp = keras.Input(shape=(IMG, IMG, 3))
    x = augment(inp)
    x = base(x, training=False)      # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out), base
""")

code("""
# ============ METRICS ============
def roc_auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p), float); sp = p[o]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j+1] == sp[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float("nan") if not (n1 and n0) else (r[y==1].sum() - n1*(n1+1)/2)/(n1*n0)

def at_threshold(y, p, t):
    yh = (p >= t).astype(int)
    tp = int(((yh==1)&(y==1)).sum()); tn = int(((yh==0)&(y==0)).sum())
    fp = int(((yh==1)&(y==0)).sum()); fn = int(((yh==0)&(y==1)).sum())
    rec = tp/(tp+fn) if tp+fn else 0.0
    spec = tn/(tn+fp) if tn+fp else 0.0
    prec = tp/(tp+fp) if tp+fp else 0.0
    return dict(threshold=float(t), accuracy=(tp+tn)/len(y),
                balanced_accuracy=(rec+spec)/2, recall=rec, specificity=spec,
                precision=prec, tp=tp, tn=tn, fp=fp, fn=fn)

def best_threshold(y, p, target_recall=0.90):
    cand = [at_threshold(y, p, t) for t in np.arange(0.02, 0.99, 0.01)]
    ok = [m for m in cand if m["recall"] >= target_recall]
    return max(ok if ok else cand, key=lambda m: m["balanced_accuracy"])
""")

code("""
# ============ TRAIN ONE FOLD ============
def train_fold(tr_df, va_df, e1=15, e2=25, verbose=2):
    tr, va = make_ds(tr_df, True), make_ds(va_df, False)
    assert_input_range(tr, "train")
    n1 = int(tr_df.label.sum()); n0 = len(tr_df) - n1
    cw = {0: len(tr_df)/(2*n0), 1: len(tr_df)/(2*n1)}
    M = [keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")]
    cbs = lambda: [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                      patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                          factor=0.3, patience=4, verbose=0)]
    model, base = build_model()
    model.compile(keras.optimizers.Adam(1e-3),
                  keras.losses.BinaryCrossentropy(), metrics=M)
    model.fit(tr, validation_data=va, epochs=e1, class_weight=cw,
              callbacks=cbs(), verbose=verbose)
    base.trainable = True
    for l in base.layers[:-40]:
        l.trainable = False
    model.compile(keras.optimizers.Adam(1e-5),
                  keras.losses.BinaryCrossentropy(), metrics=M)
    model.fit(tr, validation_data=va, epochs=e2, class_weight=cw,
              callbacks=cbs(), verbose=verbose)
    return model, model.predict(va, verbose=0).ravel(), va_df.label.values.astype(int)
""")

code("""
# ============ 5-FOLD CV ============
FOLDS = [f"fold{i}" for i in range(5)]
dev = df[df.split != "test"]

oof_p, oof_idx, scores = [], [], []
for f in FOLDS:
    print("=" * 20, f, "=" * 20)
    tr, va = dev[dev.split != f], dev[dev.split == f]
    model, p, y = train_fold(tr, va, verbose=0)
    m = best_threshold(y, p); m["auc"] = roc_auc(y, p); m["fold"] = f
    scores.append(m); oof_p.append(p); oof_idx.append(va.index.values)
    print(f"{f}: AUC={m['auc']:.4f} bal={m['balanced_accuracy']:.4f} "
          f"rec={m['recall']:.4f} spec={m['specificity']:.4f}")
    del model; keras.backend.clear_session()

oof_p = np.concatenate(oof_p); oof_idx = np.concatenate(oof_idx)
oof = dev.loc[oof_idx].copy(); oof["p"] = oof_p
oof_y = oof.label.values.astype(int)

print()
print("=" * 54)
print("v1.1 CROP MODEL -- 5-FOLD CV")
for k in ["auc", "balanced_accuracy", "recall", "specificity", "precision"]:
    v = np.array([s[k] for s in scores])
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    print(f"  {k:20} {v.mean():.4f} +/- {v.std(ddof=1):.4f}   95% CI +/-{ci:.4f}")
""")

code("""
# ============ THE ANSWER: SMALL-LESION RECALL ============
TAU = best_threshold(oof_y, oof_p)["threshold"]
print(f"operating threshold tau = {TAU:.2f}")
print()

posx = oof[oof.label == 1].copy()
qs = posx["box_area"].quantile([0.25, 0.50, 0.75]).values
bands = [("Q1 smallest", posx.box_area <= qs[0]),
         ("Q2", (posx.box_area > qs[0]) & (posx.box_area <= qs[1])),
         ("Q3", (posx.box_area > qs[1]) & (posx.box_area <= qs[2])),
         ("Q4 largest", posx.box_area > qs[2])]
print("RECALL BY LESION SIZE  (Q1 = the v1.1 target)")
for name, mask in bands:
    g = posx[mask]
    r = (g.p.values >= TAU).mean()
    se = math.sqrt(r*(1-r)/len(g)) if len(g) else 0
    print(f"  {name:12} n={len(g):5d}  recall={r:.4f}  95% CI +/-{1.96*se:.4f}")

print()
print("PER SOURCE (negatives come from both datasets):")
for s, g in oof.groupby("source"):
    yy = g.label.values.astype(int)
    if yy.min() == yy.max():
        print(f"  {s:10} n={len(g):5d}  single-class, AUC undefined")
        continue
    print(f"  {s:10} n={len(g):5d}  AUC={roc_auc(yy, g.p.values):.4f}")

print()
print("REFERENCE -- v1.0.0 whole-image model:")
print("  CV balanced accuracy 0.9148 +/- 0.0078")
print("  test Mild recall     0.8511 +/- 0.1018   <- the number to beat")
print("  NOTE: box-area quartile is a PROXY for clinical severity, not the")
print("        same measurement. A small box can sit on a severely affected")
print("        animal. Treat Q1 recall as 'focal/subtle lesion' recall.")
""")

code("""
# ============ FINAL MODEL + GRAD-CAM ============
train_part = dev[dev.split != "fold0"]
val_part   = dev[dev.split == "fold0"]
assert not (set(train_part.group_id) & set(val_part.group_id)), "overlap"
final_model, _, _ = train_fold(train_part, val_part, verbose=2)
final_model.save("/kaggle/working/bi-lsd-crop-v1.1.0.keras")

seq = [l for l in final_model.layers
       if not isinstance(l, keras.layers.InputLayer) and l.name != "augment"]
g_in = keras.Input(shape=(IMG, IMG, 3)); x = g_in; feat = None
for l in seq:
    x = l(x)
    if len(x.shape) == 4:
        feat = x
grad_model = keras.Model(g_in, [feat, x])

def gradcam(b):
    with tf.GradientTape() as tape:
        conv, pred = grad_model(b, training=False)
        tape.watch(conv); loss = pred[:, 0]
    gr = tape.gradient(loss, conv)
    w = tf.reduce_mean(gr, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))
    return (cam / (tf.reduce_max(cam, axis=(1,2), keepdims=True) + 1e-8)).numpy()

# sample SMALL lesions specifically -- that is what v1.1 exists to fix
small_pos = dev[(dev.label == 1) & (dev.box_area <= Q1)].sample(30, random_state=SEED)
imgs = np.stack([_load(p, 1)[0].numpy() for p in small_pos.path])
cams = gradcam(tf.constant(imgs))
preds = final_model.predict(imgs, verbose=0).ravel()

fig, axes = plt.subplots(5, 6, figsize=(18, 15))
for ax, im, cam, pr in zip(axes.ravel(), imgs, cams, preds):
    heat = tf.image.resize(cam[..., None], (IMG, IMG)).numpy().squeeze()
    ax.imshow(im.astype("uint8")); ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(f"p={pr:.2f}", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("/kaggle/working/gradcam_v11_small.png", dpi=110)
print("saved gradcam_v11_small.png -- SMALL lesions only")
""")

code("""
# ============ EXPORT ============
conv = tf.lite.TFLiteConverter.from_keras_model(final_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl = conv.convert()
open("/kaggle/working/bi-lsd-crop-v1.1.0.tflite", "wb").write(tfl)
print(f"tflite: {len(tfl)/1e6:.2f} MB")

res = {
  "modelVersion": "bi-lsd-crop-v1.1.0",
  "unit": "lesion crop (box + 25% context) -- NOT a whole-image classifier",
  "inputContract": "raw [0,255] RGB 224x224x3 NHWC",
  "outputContract": "[1,1] sigmoid probability",
  "tau": float(TAU),
  "cv": {k: [float(s[k]) for s in scores]
         for k in ["auc","balanced_accuracy","recall","specificity"]},
  "small_lesion_cutoff_area": float(Q1),
  "recall_by_size": {name: float((posx[mask].p.values >= TAU).mean())
                     for name, mask in bands},
  "n_by_size": {name: int(mask.sum()) for name, mask in bands},
  "test_set_evaluated": False,
  "note": "test split deliberately untouched; promote only if CV beats v1.0.0",
}
json.dump(res, open("/kaggle/working/metrics_v11.json", "w"), indent=2)
print(json.dumps(res["recall_by_size"], indent=2))
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

OUTDIR.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "hanzlaiqbal/bovine-lsd-v1-1-crops",
    "title": "bovine-lsd-v1-1-crops",
    "code_file": "bovine_lsd_v1_1.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["hanzlaiqbal/bovine-lsd-crops"],
    "competition_sources": [],
    "kernel_sources": [],
}
(OUTDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
print(f"wrote {NB}  ({len(CELLS)} cells)")
print(f"wrote {OUTDIR / 'kernel-metadata.json'}")
