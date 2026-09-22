"""Generate the Kaggle training notebook (.ipynb).

Writing .ipynb JSON by hand is escape-error-prone, so cells are authored here as
plain strings and json.dump handles the encoding.

    python ai/scripts/build_notebook.py
    -> ai/notebooks/bovine_lsd_training.ipynb
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ai" / "notebooks" / "bovine_lsd_training.ipynb"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ----------------------------------------------------------------- 0
md("""
# BovineInsight — LSD Screening Model (Phase B)

Retrains the screening model from scratch. The previous model
(`finals_models00.tflite`) failed Phase A: **0.6253 balanced accuracy, below the
0.6836 majority baseline**, because it was trained on `[0,1]`-normalised images
fed into a network that already contained `include_preprocessing=True`.

**Every guard in this notebook exists to stop that recurring.**

## Before you run

1. **Settings → Accelerator → GPU T4 ×2** (needs phone verification)
2. **Add Data →** upload `bovine_lsd_bundle.zip` as a Kaggle Dataset
3. Set `BUNDLE` below to its path

## What the bundle contains

3,030 images at 256px, already deduplicated and split.

| | |
|---|---|
| True dataset size | 2,798 groups (not 3,030 files) |
| Lesion / Normal | 1,833 / 965 groups |
| Test set | 559 groups — **LOCKED** |
| CV folds | 5 × ~448 groups |

Splits are **group-aware** (near-duplicates never straddle folds) and stratified
on the **joint (label, source)**, because source nearly determines label:
`P(lesion | mendeley) = 0.299` vs `P(lesion | roboflow) = 0.826`.
""")

# ----------------------------------------------------------------- 1
code("""
# ============ SETUP: seed everything, verify GPU ============
import os, random, json, math, io, zipfile, shutil
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras import layers

print("TF", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
assert gpus, "No GPU. Settings -> Accelerator -> GPU T4 x2 (needs phone verification)."
""")

# ----------------------------------------------------------------- 2
code("""
# ============ LOAD BUNDLE ============
# Kaggle AUTO-EXTRACTS uploaded .zip datasets, so there may be no zip to open.
# Find manifest.csv wherever it landed; fall back to extracting a zip.
import glob
import pandas as pd

for root, dirs, files in os.walk("/kaggle/input"):
    if root.count("/") <= 4:
        print(root, f"-> {len(dirs)} dirs, {len(files)} files")

hits = glob.glob("/kaggle/input/**/manifest.csv", recursive=True)
if hits:
    WORK = os.path.dirname(hits[0])
    print("\\nusing pre-extracted dataset at", WORK)
else:
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    assert zips, "No manifest.csv and no .zip under /kaggle/input -- dataset attached?"
    WORK = "/kaggle/working/data"
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
    print("\\nextracted", zips[0], "->", WORK)

df = pd.read_csv(f"{WORK}/manifest.csv")
df["path"] = WORK + "/" + df["file"]
df["label"] = df["label"].astype(int)
assert os.path.exists(df["path"].iloc[0]), f"images missing: {df['path'].iloc[0]}"
print("\\nshape:", df.shape)

print("\\nBY SPLIT:")
print(df.groupby("split")["label"].agg(n="size", lesion="sum",
      p_lesion="mean").round(3))
print("\\nBY SOURCE:")
print(df.groupby("source")["label"].agg(n="size", p_lesion="mean").round(3))
print("\\nSEVERITY:", df["severity"].value_counts(dropna=False).to_dict())

# sanity: the split assignment must already be group-safe
assert df.groupby("group_id")["split"].nunique().max() == 1, \\
    "GROUP LEAK: a group spans multiple splits"
print("\\n[PASS] no group spans multiple splits")
""")

# ----------------------------------------------------------------- 3
code("""
# ============ DATA PIPELINE ============
# CRITICAL: pixels stay in [0,255]. MobileNetV3 with include_preprocessing=True
# rescales internally (x/127.5 - 1). Normalising here is the bug that killed v0.
IMG, BATCH = 224, 32
AUTO = tf.data.AUTOTUNE

def _load(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG, IMG))
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)   # stays 0-255

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.3, value_range=(0, 255)),   # NOT the (0,1) default
    layers.RandomContrast(0.3),
], name="augment")

def _gamma(img, label):
    \"\"\"Dark-hide robustness. Pakistan's herd is ~59% black-hided buffalo;
    every public dataset here is light-coated cattle.\"\"\"
    g = tf.random.uniform([], 0.5, 1.8)
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

# ----------------------------------------------------------------- 4
code("""
# ============ THE GUARD ============
# One line. Would have prevented the entire v0 failure.
def assert_input_range(ds, name):
    xb, _ = next(iter(ds))
    lo, hi = float(tf.reduce_min(xb)), float(tf.reduce_max(xb))
    assert hi > 1.5, (
        f"INPUT RANGE BUG in {name}: max={hi:.4f}. "
        "MobileNetV3(include_preprocessing=True) expects RAW [0,255]. "
        "This is exactly what killed finals_models00.tflite."
    )
    print(f"[PASS] {name} input range [{lo:.1f}, {hi:.1f}]")

_probe = make_ds(df[df.split == "fold0"], training=True)
assert_input_range(_probe, "probe")
""")

# ----------------------------------------------------------------- 5
code("""
# ============ MODEL ============
def build_model():
    base = keras.applications.MobileNetV3Large(
        input_shape=(IMG, IMG, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,   # GOTCHA 1: wants raw [0,255]
    )
    base._name = "backbone"   # Keras 3 may ignore this; Grad-CAM finds it by shape
    base.trainable = False

    inp = keras.Input(shape=(IMG, IMG, 3))
    x = augment(inp)
    # GOTCHA 2: training=False keeps BatchNorm in inference mode. BN holds two
    # NON-trainable weights (running mean/var) that keep updating even when
    # base.trainable=False. Letting them restatistic on batches of 32 destroys
    # the ImageNet features. See keras.io/guides/transfer_learning.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # 1-unit head = V0 contract
    return keras.Model(inp, out), base

_m, _b = build_model()
print(f"params: {_m.count_params():,}   output: {_m.output_shape}")
assert _m.output_shape[-1] == 1, "expected a 1-unit sigmoid head"
del _m, _b
""")

# ----------------------------------------------------------------- 6
code("""
# ============ METRICS (no sklearn dependency) ============
def roc_auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p), float); sp = p[o]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j+1] == sp[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float("nan") if not (n1 and n0) else \\
        (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)

def at_threshold(y, p, t):
    yh = (p >= t).astype(int)
    tp = int(((yh==1)&(y==1)).sum()); tn = int(((yh==0)&(y==0)).sum())
    fp = int(((yh==1)&(y==0)).sum()); fn = int(((yh==0)&(y==1)).sum())
    rec  = tp/(tp+fn) if tp+fn else 0.0
    spec = tn/(tn+fp) if tn+fp else 0.0
    prec = tp/(tp+fp) if tp+fp else 0.0
    return dict(threshold=float(t), accuracy=(tp+tn)/len(y),
                balanced_accuracy=(rec+spec)/2, recall=rec, specificity=spec,
                precision=prec, f1=2*prec*rec/(prec+rec) if prec+rec else 0.0,
                tp=tp, tn=tn, fp=fp, fn=fn)

def best_threshold(y, p, target_recall=0.90):
    \"\"\"Asymmetric by design: a false negative tells a farmer a diseased animal
    is fine. Hit the recall target first, then maximise balanced accuracy.\"\"\"
    cand = [at_threshold(y, p, t) for t in np.arange(0.02, 0.99, 0.01)]
    ok = [m for m in cand if m["recall"] >= target_recall]
    pool = ok if ok else cand
    return max(pool, key=lambda m: m["balanced_accuracy"])
""")

# ----------------------------------------------------------------- 7
code("""
# ============ TRAIN ONE FOLD (two stages) ============
def train_fold(train_df, val_df, epochs1=15, epochs2=25, verbose=2):
    tr = make_ds(train_df, training=True)
    va = make_ds(val_df,  training=False)
    assert_input_range(tr, "train")

    n1 = int(train_df.label.sum()); n0 = len(train_df) - n1
    cw = {0: len(train_df)/(2*n0), 1: len(train_df)/(2*n1)}

    METRICS = [keras.metrics.AUC(name="auc"),
               keras.metrics.AUC(name="pr_auc", curve="PR"),
               keras.metrics.BinaryAccuracy(name="acc")]
    cbs = lambda: [
        # val_auc, NOT val_acc -- "always lesion" already scores 0.655 here
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                      patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                          factor=0.3, patience=4, verbose=0),
    ]

    model, base = build_model()

    # --- stage 1: head only ---
    model.compile(keras.optimizers.Adam(1e-3),
                  keras.losses.BinaryCrossentropy(), metrics=METRICS)
    model.fit(tr, validation_data=va, epochs=epochs1,
              class_weight=cw, callbacks=cbs(), verbose=verbose)

    # --- stage 2: unfreeze the top of the backbone ---
    base.trainable = True
    for l in base.layers[:-40]:
        l.trainable = False
    # MUST recompile after changing trainability, and drop LR 100x
    model.compile(keras.optimizers.Adam(1e-5),
                  keras.losses.BinaryCrossentropy(), metrics=METRICS)
    model.fit(tr, validation_data=va, epochs=epochs2,
              class_weight=cw, callbacks=cbs(), verbose=verbose)

    p = model.predict(va, verbose=0).ravel()
    return model, p, val_df.label.values.astype(int)
""")

# ----------------------------------------------------------------- 8
code("""
# ============ SOURCE-ADVERSARIAL CHECK (run this BEFORE trusting any score) ==
# If a model can tell mendeley from roboflow almost perfectly, then "guess the
# dataset" is an available shortcut to ~80% on the real task, and any headline
# accuracy is suspect until Grad-CAM proves otherwise.
adv = df[df.split != "test"].copy()
adv["label"] = (adv["source"] == "roboflow").astype(int)   # predict SOURCE
a_tr = adv[adv.split != "fold0"]; a_va = adv[adv.split == "fold0"]

_m, _ = build_model()
_m.compile(keras.optimizers.Adam(1e-3), keras.losses.BinaryCrossentropy(),
           metrics=[keras.metrics.AUC(name="auc")])
_m.fit(make_ds(a_tr, True), validation_data=make_ds(a_va, False),
       epochs=5, verbose=2)
src_auc = roc_auc(a_va.label.values.astype(int),
                  _m.predict(make_ds(a_va, False), verbose=0).ravel())
print(f"\\n>>> SOURCE-PREDICTION AUC = {src_auc:.4f}")
print("    >0.95 : the shortcut is trivially available. Grad-CAM (Step 9)")
print("            now DECIDES whether this model ships.")
del _m
""")

# ----------------------------------------------------------------- 9
code("""
# ============ 5-FOLD CROSS-VALIDATION ============
FOLDS = [f"fold{i}" for i in range(5)]
dev = df[df.split != "test"]

oof_p, oof_y, oof_idx, fold_scores = [], [], [], []
for f in FOLDS:
    print(f"\\n{'='*22} {f} {'='*22}")
    tr = dev[dev.split != f]; va = dev[dev.split == f]
    model, p, y = train_fold(tr, va, verbose=0)
    m = best_threshold(y, p)
    m["auc"] = roc_auc(y, p); m["fold"] = f
    fold_scores.append(m)
    oof_p.append(p); oof_y.append(y); oof_idx.append(va.index.values)
    print(f"{f}: AUC={m['auc']:.4f}  bal={m['balanced_accuracy']:.4f}  "
          f"rec={m['recall']:.4f}  spec={m['specificity']:.4f}")
    del model; keras.backend.clear_session()

oof_p = np.concatenate(oof_p); oof_y = np.concatenate(oof_y)
oof_idx = np.concatenate(oof_idx)

print("\\n" + "="*54)
print("5-FOLD CV  (mean +/- std, 95% CI)")
for k in ["auc", "balanced_accuracy", "recall", "specificity", "precision"]:
    v = np.array([s[k] for s in fold_scores])
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    print(f"  {k:20} {v.mean():.4f} +/- {v.std(ddof=1):.4f}   95% CI +/-{ci:.4f}")
print(f"\\n  majority baseline (always lesion): {dev.label.mean():.4f}")
print(f"  old dead model:                     0.6253")
""")

# ----------------------------------------------------------------- 10
code("""
# ============ STRATIFIED EVALUATION — the honest part ============
oof = dev.loc[oof_idx].copy()
oof["p"] = oof_p
TAU = best_threshold(oof_y, oof_p)["threshold"]
print(f"operating threshold tau = {TAU:.2f}\\n")

print("PER SOURCE  (aggregate numbers hide the shortcut):")
for s, g in oof.groupby("source"):
    y = g.label.values.astype(int); p = g.p.values
    m = at_threshold(y, p, TAU)
    print(f"  {s:10} n={len(g):5d}  AUC={roc_auc(y,p):.4f}  "
          f"bal={m['balanced_accuracy']:.4f}  rec={m['recall']:.4f}")

print("\\nPER SEVERITY  (does it catch EARLY disease, or only obvious disease?):")
sev = oof[oof.severity.notna()]
for s in ["Mild", "Severe"]:
    g = sev[sev.severity == s]
    if not len(g): continue
    det = (g.p.values >= TAU).sum()
    n = len(g)
    se = math.sqrt(det/n*(1-det/n)/n) if n else 0
    print(f"  {s:8} n={n:4d}  recall={det/n:.4f}  95% CI +/-{1.96*se:.4f}")
print("\\n  >>> MILD recall is the early-detection claim. If Mild recall is far")
print("      below Severe, the model only sees what the farmer already sees.")
""")

# ----------------------------------------------------------------- 10b
code("""
# ============ THRESHOLD SWEEP: tuned on MILD recall ============
# Mild ~= early-stage disease -- where the product's value actually lives.
# A false negative can cost a herd; a false positive costs a vet visit.
_sev  = oof[oof.severity.notna()]
_mild = _sev[_sev.severity == "Mild"]
_svr  = _sev[_sev.severity == "Severe"]
print(f"Mild n={len(_mild)}   Severe n={len(_svr)}")

print(f"{'tau':>6} {'Mild':>8} {'Severe':>8} {'recall':>8} {'spec':>8} {'bal':>8} {'prec':>8}")
print("-" * 60)
rows = []
for t in np.arange(0.20, 0.96, 0.05):
    m = at_threshold(oof_y, oof_p, t)
    mild_r = (_mild.p >= t).mean(); svr_r = (_svr.p >= t).mean()
    rows.append((t, mild_r, svr_r, m))
    print(f"{t:6.2f} {mild_r:8.4f} {svr_r:8.4f} {m['recall']:8.4f} "
          f"{m['specificity']:8.4f} {m['balanced_accuracy']:8.4f} {m['precision']:8.4f}")

# highest tau still reaching 85% Mild recall = least specificity sacrificed
ok = [r for r in rows if r[1] >= 0.85]
print()
if ok:
    t, mild_r, svr_r, m = max(ok, key=lambda r: r[0])
    print(f">>> Mild recall >= 0.85 reachable at tau = {t:.2f}")
    print(f"    Mild={mild_r:.4f}  Severe={svr_r:.4f}  spec={m['specificity']:.4f}  "
          f"bal={m['balanced_accuracy']:.4f}")
    base = at_threshold(oof_y, oof_p, TAU)
    print(f"    cost vs tau={TAU:.2f}: specificity {base['specificity']:.4f} -> "
          f"{m['specificity']:.4f}  ({m['specificity']-base['specificity']:+.4f})")
else:
    print(">>> Mild recall NEVER reaches 0.85 at any threshold.")
    print("    The model cannot resolve early lesions. State this as a scope")
    print("    limit in the model card; do not tune around it.")
""")

# ----------------------------------------------------------------- 11
code("""
# ============ CALIBRATION (Guo et al. 2017) ============
# The head is a 1-unit SIGMOID, so recover the logit before temperature scaling.
from scipy.optimize import minimize_scalar

def inv_sigmoid(p, eps=1e-7):
    p = np.clip(p, eps, 1-eps); return np.log(p/(1-p))

def nll(T, logit, y):
    q = 1/(1+np.exp(-logit/T)); q = np.clip(q, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(q) + (1-y)*np.log(1-q))

def ece(p, y, bins=15):
    \"\"\"Binary ECE: |observed frequency - mean predicted probability| per bin.

    NOT accuracy-vs-confidence. For a 1-unit sigmoid head the question that
    matters is "does p=0.7 mean 70% of these are lesions?", because that is
    exactly what the Gate 3 threshold relies on. Comparing bin ACCURACY to
    mean p instead makes every confidently-correct negative (p=0.05, y=0)
    contribute |1.0 - 0.05| and inflates the score to meaninglessness.
    \"\"\"
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

logit = inv_sigmoid(oof_p)
T = minimize_scalar(nll, bounds=(0.05,10.0), args=(logit,oof_y),
                    method="bounded").x
cal = 1/(1+np.exp(-logit/T))
print(f"temperature T = {T:.4f}")
print(f"ECE before = {ece(oof_p,oof_y):.4f}")
print(f"ECE after  = {ece(cal,oof_y):.4f}    (acceptance bar: <= 0.10)")
TAU_CAL = best_threshold(oof_y, cal)["threshold"]
print(f"calibrated threshold tau = {TAU_CAL:.2f}")
""")

# ----------------------------------------------------------------- 12
code("""
# ============ FINAL MODEL ============
# Validation MUST be disjoint from training or early stopping is meaningless:
# monitoring val_auc on data the model is memorising means it never fires.
# Train on folds 1-4, hold out fold0 purely as the stopping signal.
# Costs 20% of dev; buys a valid convergence criterion.
train_part = dev[dev.split != "fold0"]
val_part   = dev[dev.split == "fold0"]
assert not (set(train_part.group_id) & set(val_part.group_id)), "train/val overlap"
print(f"train {len(train_part)}  val {len(val_part)}  (disjoint groups)")

final_model, _, _ = train_fold(train_part, val_part, verbose=2)
final_model.save("/kaggle/working/bi-lsd-mnv3l-v1.0.0.keras")
print("saved final model")
""")

# ----------------------------------------------------------------- 13
code("""
# ============ GRAD-CAM AUDIT — can veto a model that passed everything ======
# Web-sourced data: healthy cows skew to clean stock photography, diseased cows
# come from news/vet reports. A model can score 95% on photo STYLE. A confusion
# matrix will never show it. Look at the pictures.
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# name-independent: Keras 3 ignores base._name, so find the last 4-D
# feature map before pooling rather than hardcoding a layer name.
bb = [l for l in final_model.layers
      if hasattr(l, "output") and len(getattr(l.output, "shape", ())) == 4][-1]
print("backbone layer:", bb.name, bb.output.shape)
grad_model = keras.Model(final_model.inputs, [bb.output, final_model.output])

def gradcam(batch):
    with tf.GradientTape() as tape:
        conv, pred = grad_model(batch, training=False)
        tape.watch(conv)
        loss = pred[:, 0]
    grads = tape.gradient(loss, conv)
    w = tf.reduce_mean(grads, axis=(1,2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w*conv, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1,2), keepdims=True) + 1e-8)
    return cam.numpy()

pool = dev[dev.label == 1].sample(30, random_state=SEED)
imgs = np.stack([_load(p, 1)[0].numpy() for p in pool.path])
cams = gradcam(tf.constant(imgs))
preds = final_model.predict(imgs, verbose=0).ravel()

fig, axes = plt.subplots(5, 6, figsize=(18, 15))
for ax, im, cam, pr in zip(axes.ravel(), imgs, cams, preds):
    heat = tf.image.resize(cam[..., None], (IMG, IMG)).numpy().squeeze()
    ax.imshow(im.astype("uint8"))
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(f"p={pr:.2f}", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig("/kaggle/working/gradcam.png", dpi=110)
plt.show()
print(">>> PASS if >=25/30 highlight SKIN / LESIONS.")
print(">>> FAIL if activation sits on background, fences, ear tags, watermarks,")
print("    or image borders -- then the accuracy is an artifact. Fix the DATA.")
""")

# ----------------------------------------------------------------- 14
code("""
# ============ THE LOCKED TEST SET — RUN EXACTLY ONCE ============
# Tuning anything after seeing this turns it into a validation set and you will
# need a fresh one. Flip the flag only when you are finished.
RUN_TEST = False
assert RUN_TEST, "Set RUN_TEST=True only when all tuning is complete."

test = df[df.split == "test"]
tp_ = final_model.predict(make_ds(test, False), verbose=0).ravel()
ty_ = test.label.values.astype(int)
tp_cal = 1/(1+np.exp(-inv_sigmoid(tp_)/T))

m = at_threshold(ty_, tp_cal, TAU_CAL)
print("TEST SET (once):")
print(f"  AUC   {roc_auc(ty_, tp_cal):.4f}")
print(f"  bal   {m['balanced_accuracy']:.4f}   rec {m['recall']:.4f}   "
      f"spec {m['specificity']:.4f}")
print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
for s, g in test.assign(p=tp_cal).groupby("source"):
    print(f"  {s:10} AUC={roc_auc(g.label.values.astype(int), g.p.values):.4f}")
""")

# ----------------------------------------------------------------- 15
code("""
# ============ EXPORT + PARITY CHECK ============
conv = tf.lite.TFLiteConverter.from_keras_model(final_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl = conv.convert()
open("/kaggle/working/bi-lsd-mnv3l-v1.0.0.tflite", "wb").write(tfl)
print(f"tflite size: {len(tfl)/1e6:.2f} MB   (bar: <= 8 MB)")

# Quantisation shifts decision boundaries. Verify BEFORE trusting thresholds.
sample = dev.sample(100, random_state=SEED)
xs = np.stack([_load(p, 0)[0].numpy() for p in sample.path])
keras_p = final_model.predict(xs, verbose=0).ravel()

itp = tf.lite.Interpreter(model_content=tfl); itp.allocate_tensors()
i_d, o_d = itp.get_input_details()[0], itp.get_output_details()[0]
print("tflite input :", i_d["shape"], i_d["dtype"].__name__)
print("tflite output:", o_d["shape"], o_d["dtype"].__name__)
tfl_p = []
for x in xs:
    itp.set_tensor(i_d["index"], x[None].astype(np.float32)); itp.invoke()
    tfl_p.append(float(itp.get_tensor(o_d["index"])[0][0]))
tfl_p = np.array(tfl_p)

d = np.abs(keras_p - tfl_p).max()
print(f"\\nmax |keras - tflite| = {d:.6f}   (bar: < 1e-3)")
ys = sample.label.values.astype(int)
print(f"AUC keras  = {roc_auc(ys, keras_p):.4f}")
print(f"AUC tflite = {roc_auc(ys, tfl_p):.4f}")
print("\\n>>> If these diverge, RE-RUN calibration on the TFLITE outputs and ship")
print("    those thresholds. Always calibrate the artifact you deploy.")

json.dump({
    "modelVersion": "bi-lsd-mnv3l-v1.0.0",
    "inputContract": "raw [0,255] RGB, 224x224x3, NHWC",
    "outputContract": "[1,1] sigmoid probability of lesion",
    "temperature": float(T), "tau": float(TAU_CAL),
    "cv": {k: [float(s[k]) for s in fold_scores]
           for k in ["auc","balanced_accuracy","recall","specificity"]},
    "sourcePredictionAUC": float(src_auc),
    "keras_tflite_max_diff": float(d),
}, open("/kaggle/working/metrics.json","w"), indent=2)
print("\\nwrote metrics.json")
""")

# ----------------------------------------------------------------- 16
md("""
## Acceptance gate

Download from `/kaggle/working/`: the `.tflite`, `metrics.json`, `gradcam.png`.

| Test | Bar |
|---|---|
| T2 — 5-fold CV balanced accuracy | **≥ 0.85** mean, std ≤ 0.03 |
| T5 — abstention rate | ≤ 35% |
| T7 — ECE after temperature scaling | **≤ 0.10** |
| T8 — **Grad-CAM audit** | **≥ 25/30 on skin/lesions** |
| T9 — per-source metrics | reported separately, never aggregated |
| T11 — model size | ≤ 8 MB |
| Parity — Keras vs TFLite | < 1e-3 |

**Reference points:** majority baseline 0.655 · old dead model 0.625 · honest
literature ceiling ~0.88. A result near 0.98 means you have a leak, not a
breakthrough.

### What this still cannot tell you

Every image here is web-sourced cattle. **No buffalo** — 59.4% of Pakistan's
milk, black-hided, and LSD presents differently in them (small nodules without
centred ulcerations). Clearing this gate earns you the right to ship in
**collect-only mode** and start building the field set.

**T3 — 250+ vet-labelled photos from real Pakistani farms — remains the only
number that decides production readiness.**
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(CELLS)} cells)")
