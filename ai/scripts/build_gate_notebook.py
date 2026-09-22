"""Generate the Kaggle kernel that adds Gate 1 + species detection to the model.

Field testing (2026-09-21/22) produced a verdict for a photograph of a hand.
That is the Gate 1 gap from AI_IMPLEMENTATION_PLAN.md section 6.2, and it has
to close before the link is shared any wider.

WHY TWO GATES. The first attempt was a single Dense(3) softmax head -- cattle /
buffalo / other -- on the LSD backbone's frozen features. Measured locally on
twenty real landscape photographs it rejected ZERO: every one scored
p(cattle) ~ 1.000. The fine-tuned backbone learned "field, grass, sky = cattle",
and a closed softmax has no way to say "none of these" (section 6.1 of the plan
says exactly this). A Mahalanobis distance on the same features, by contrast,
reached AUROC 0.993 non-animal-vs-animal in that same local test. So:

  output "species" -- Dense(3) softmax. Tells cattle from buffalo (99%+ both
                      ways), and catches the negative CATEGORIES it was shown.
  output "ood"     -- Mahalanobis d^2 of the 960-d pooled feature from the
                      bovine training distribution, computed IN THE GRAPH by
                      two fixed Dense layers (PCA projection, then whitening).
                      Open-set: it does not need to have seen the category.
  output "lesion"  -- unchanged, byte-for-byte the shipped weights.

Reject when d^2 >= oodMax OR p(other) >= otherMax. Both thresholds are chosen
from sweeps so that >= 99% of real cattle AND buffalo pass.

HONEST NEGATIVES. Training negatives are natural-images, real hand photos, and
70% of Caltech-101's object classes. Everything the gate is JUDGED on for
open-set behaviour was never trained on in any form: Intel scenes (buildings,
forest, glacier, mountain, sea, street), LFW faces, and the other 30% of
Caltech classes -- class-disjoint, not image-disjoint. The 'other' accuracy on
the in-domain held-out split is reported too, but it is the novel-set numbers
that describe what happens when a stranger points the camera at a wall.

The lesion output MUST be numerically identical to the shipped model. The
kernel asserts it before exporting; ai/scripts/verify_gate.py asserts it again
locally against the file that actually ships.

Kaggle returns an EMPTY log for failed runs through the CLI, so every cell
appends to progress.txt and any exception lands in error.txt.
"""
import pathlib

CELLS = []


def cell(src):
    CELLS.append(src.strip("\n"))


# ------------------------------------------------------------------ 1. env
cell(r'''
import sys, os, glob, json, subprocess, shutil, io, csv, zipfile, random, time, traceback

def note(msg):
    with open("/kaggle/working/progress.txt", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

def _excepthook(t, v, tb):
    open("/kaggle/working/error.txt", "w").write("".join(traceback.format_exception(t, v, tb)))
    note("UNCAUGHT: " + "".join(traceback.format_exception_only(t, v)).strip())
    sys.__excepthook__(t, v, tb)
sys.excepthook = _excepthook
note("1: start")

print("python     ", sys.version.split()[0])
import numpy as np;      print("numpy      ", np.__version__)
import tensorflow as tf; print("tensorflow ", tf.__version__)
import keras;            print("keras      ", keras.__version__)
print("GPUs       ", tf.config.list_physical_devices("GPU"))
for p in sorted(glob.glob("/kaggle/input/*")):
    print(p)
''')

# ------------------------------------------------------------------ 2. install
cell(r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime",
                "onnxconverter-common"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "tf2onnx"], check=False)
import importlib
for m in ("onnx", "onnxruntime", "tf2onnx", "onnxconverter_common", "tensorflow"):
    mod = importlib.import_module(m)
    print(f"{m:22} {getattr(mod, '__version__', '?')}")
note("2: installs ok")
''')

# ------------------------------------------------------------------ 3. model + feature extractor
cell(r'''
KERAS = sorted(glob.glob("/kaggle/input/**/*.keras", recursive=True))[0]
model = keras.models.load_model(KERAS, compile=False)

backbone = model.get_layer("MobileNetV3Large")
gap      = model.get_layer("global_average_pooling2d")
dense    = model.get_layer("dense")      # 128, relu
dense_1  = model.get_layer("dense_1")    # 1, sigmoid

inp    = keras.Input(shape=(224, 224, 3), dtype="float32", name="image")
feat   = gap(backbone(inp))              # 960-d pooled features, shared by all heads
lesion = dense_1(dense(feat))
infer  = keras.Model(inp, lesion, name="bi_lsd_infer")
feature_extractor = keras.Model(inp, feat, name="bi_lsd_features")

rng = np.random.default_rng(42)
probe = rng.uniform(0, 255, size=(8, 224, 224, 3)).astype(np.float32)
d = float(np.max(np.abs(model.predict(probe, verbose=0) - infer.predict(probe, verbose=0))))
assert d < 1e-5, d
note(f"3: inference rebuild == original (max diff {d:.1e})")
''')

# ------------------------------------------------------------------ 4. data
cell(r'''
from PIL import Image, ImageOps
from collections import Counter

# ---- manifests by CONTENT: cattle has `split`, buffalo has `breed` ---------
manifests = glob.glob("/kaggle/input/**/manifest.csv", recursive=True)
def header(p):
    with open(p, newline="") as f:
        return set(next(csv.reader(f)))
cattle_manifest = next(p for p in manifests if {"split", "group_id"} <= header(p))
buf_manifest    = next(p for p in manifests if "breed" in header(p))
cattle_root, buf_root = os.path.dirname(cattle_manifest), os.path.dirname(buf_manifest)
cattle_rows = list(csv.DictReader(open(cattle_manifest)))
buf_rows    = list(csv.DictReader(open(buf_manifest)))
seen, buf_unique = set(), []
for r in buf_rows:
    if r["sha256"] not in seen:
        seen.add(r["sha256"]); buf_unique.append(r)
note(f"4: cattle {len(cattle_rows)}  buffalo {len(buf_unique)}")

def jpgs(root):
    return sorted(p for p in glob.glob(root + "/**/*", recursive=True)
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))

# ---- negatives. Class = parent directory name. --------------------------------
nat = {"/".join(p.split("/")[-2:]): p for p in jpgs("/kaggle/input/natural-images")}   # shipped twice; dedupe
nat = sorted(nat.values())
hands = jpgs("/kaggle/input/rock-paper-scissors")
caltech = jpgs("/kaggle/input/caltech-101")
scenes  = jpgs("/kaggle/input/intel-image-classification")
faces   = jpgs("/kaggle/input/lfw-dataset")
note(f"4: natural {len(nat)}  hands {len(hands)}  caltech {len(caltech)}  scenes {len(scenes)}  faces {len(faces)}")

rnd = random.Random(42)
rnd.shuffle(hands);  hands  = hands[:1500]
rnd.shuffle(scenes); scenes = scenes[:3000]
rnd.shuffle(faces);  faces  = faces[:1000]

# Caltech-101: CLASS-disjoint split. ~70% of classes train (+ in-domain test);
# ~30% are never seen in any form and form part of the NOVEL set.
cal_by_class = {}
for p in caltech:
    cal_by_class.setdefault(p.split("/")[-2], []).append(p)
cal_classes = sorted(cal_by_class)
novel_classes = set(cal_classes[2::3])          # every third class, deterministic
note(f"4: caltech classes {len(cal_classes)}  novel classes {len(novel_classes)}")

# ---- assemble items: cls 0 cattle, 1 buffalo, 2 other; split train/test/novel
items = []
for r in cattle_rows:
    items.append(dict(kind="cattle", ref=os.path.join(cattle_root, r["file"]), cls=0,
                      split="test" if r["split"] == "test" else "train", breed="", source=r["source"]))
by_breed = {}
for r in buf_unique:
    by_breed.setdefault(r["breed"], []).append(r)
for breed, rows in by_breed.items():
    rnd.shuffle(rows)
    n_test = max(1, round(0.25 * len(rows)))
    for i, r in enumerate(rows):
        items.append(dict(kind="buffalo", ref=os.path.join(buf_root, r["file"]), cls=1,
                          split="test" if i < n_test else "train", breed=breed, source="buffalo_pak"))

def add_other(paths, source, split_fn):
    for p in paths:
        items.append(dict(kind="other", ref=p, cls=2, split=split_fn(p), breed="", source=source))

def split80(_):
    return "test" if rnd.random() < 0.2 else "train"

add_other(nat,   "natural",  split80)
add_other(hands, "hands",    split80)
for c, paths in cal_by_class.items():
    if c in novel_classes:
        add_other(paths, "caltech_novel", lambda _: "novel")
    else:
        add_other(paths, "caltech", split80)
add_other(scenes, "scenes", lambda _: "novel")
add_other(faces,  "faces",  lambda _: "novel")

note("4: counts " + json.dumps({f"{c}/{s}": n for (c, s), n in
     sorted(Counter((it["cls"], it["split"]) for it in items).items())}))
note("4: novel sources " + json.dumps(dict(Counter(it["source"] for it in items if it["split"] == "novel"))))

def load_image(path):
    im = Image.open(path)
    # EXIF-upright first: that is what the browser hands the model.
    im = ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32)   # RAW [0,255]
''')

# ------------------------------------------------------------------ 5. features
cell(r'''
t0 = time.time()
BATCH = 64
feats = np.zeros((len(items), 960), np.float32)
lesion_p = np.zeros(len(items), np.float32)
bad = 0
for s in range(0, len(items), BATCH):
    chunk = items[s:s+BATCH]
    imgs = []
    for it in chunk:
        try:
            imgs.append(load_image(it["ref"]))
        except Exception:
            imgs.append(np.zeros((224, 224, 3), np.float32)); it["bad"] = True; bad += 1
    batch = np.stack(imgs)
    feats[s:s+len(batch)] = feature_extractor.predict(batch, verbose=0)
    lesion_p[s:s+len(batch)] = infer.predict(batch, verbose=0).ravel()
    if (s // BATCH) % 40 == 0:
        note(f"5: {s+len(batch)}/{len(items)}  {time.time()-t0:.0f}s")
keep = np.array([not it.get("bad") for it in items])
items = [it for it, k in zip(items, keep) if k]
feats, lesion_p = feats[keep], lesion_p[keep]
note(f"5: features {feats.shape} in {time.time()-t0:.0f}s  (unreadable files dropped: {bad})")

y      = np.array([it["cls"] for it in items])
split  = np.array([it["split"] for it in items])
source = np.array([it["source"] for it in items])
breed  = np.array([it["breed"] for it in items])
tr, te, nv = split == "train", split == "test", split == "novel"

o_te, o_nv = te & (y == 2), nv & (y == 2)
note(f"5: SHIPPED lesion model on non-animals -- in-domain held-out n={int(o_te.sum())}: "
     f"flags {float((lesion_p[o_te] >= 0.5).mean()):.1%};  novel n={int(o_nv.sum())}: "
     f"flags {float((lesion_p[o_nv] >= 0.5).mean()):.1%}")
''')

# ------------------------------------------------------------------ 6. species head
cell(r'''
counts = np.bincount(y[tr], minlength=3)
cw = {c: float(len(y[tr]) / (3 * counts[c])) for c in range(3)}
idx = np.where(tr)[0]; np.random.default_rng(42).shuffle(idx)
n_val = len(idx) // 10
va_idx, fit_idx = idx[:n_val], idx[n_val:]

keras.utils.set_random_seed(42)
head = keras.Sequential([
    keras.Input(shape=(960,)),
    keras.layers.Dense(3, activation="softmax", name="species",
                       kernel_regularizer=keras.regularizers.l2(1e-4)),
], name="species_head")
head.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
hist = head.fit(feats[fit_idx], y[fit_idx], validation_data=(feats[va_idx], y[va_idx]),
                epochs=80, batch_size=128, class_weight=cw, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                                         monitor="val_loss")])
P = head.predict(feats, verbose=0)
pred = P.argmax(1)
note(f"6: head trained, epochs {len(hist.history['loss'])}, val acc {max(hist.history['val_accuracy']):.4f}")
''')

# ------------------------------------------------------------------ 7. Mahalanobis OOD
cell(r'''
# Fit on BOVINE TRAIN features only. PCA to k dims, shrinkage covariance,
# whitening matrix L so that d^2 = || (f - mu) P L ||^2 -- two fixed Dense
# layers in the graph, one scalar out.
Xb = feats[tr & (y < 2)]
mu = Xb.mean(0)
U, S, Vt = np.linalg.svd(Xb - mu, full_matrices=False)
note(f"7: OOD fit on {len(Xb)} bovine train features")

def fit_ood(k, shrink=0.05):
    Pk = Vt[:k].T.astype(np.float64)                      # 960 x k
    Z = (Xb - mu) @ Pk
    cov = np.cov(Z, rowvar=False)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / k * np.eye(k)
    L = np.linalg.cholesky(np.linalg.inv(cov))            # k x k, d2 = ||z L||^2
    return Pk, L

def d2_of(F, Pk, L):
    W = ((F - mu) @ Pk) @ L
    return np.einsum("ij,ij->i", W, W)

def auroc(pos, neg):
    return float((pos[:, None] > neg[None, :]).mean())

# Choose k on the IN-DOMAIN held-out split only (bovine test vs other test).
# The novel set is never used for any choice -- it is what we report on.
bov_te = te & (y < 2)
cands = {}
for k in (64, 128, 192, 256, 384):
    Pk, L = fit_ood(k)
    d2 = d2_of(feats, Pk, L)
    a = auroc(d2[o_te], d2[bov_te])
    cands[k] = (a, Pk, L, d2)
    note(f"7: k={k:3d}  AUROC(other-test vs bovine-test)={a:.4f}  "
         f"median d2 bovine-test {np.median(d2[bov_te]):.1f} other-test {np.median(d2[o_te]):.1f} novel {np.median(d2[o_nv]):.1f}")
K = max(cands, key=lambda k: cands[k][0])
_, P_ood, L_ood, D2 = cands[K]
note(f"7: chosen k={K}")

# Threshold sweep: keep >= X of BOTH real species; report what gets rejected.
cat_te, buf_te = te & (y == 0), te & (y == 1)
ood_sweep = []
print(f"\n  {'keep':>6} {'D':>8} {'cattle kept':>12} {'buffalo kept':>13} {'other-test rej':>15} {'NOVEL rej':>10}  by novel source")
for keep_frac in (0.98, 0.99, 0.995, 0.999):
    D = float(max(np.percentile(D2[cat_te], 100 * keep_frac), np.percentile(D2[buf_te], 100 * keep_frac)))
    row = {"keep_target": keep_frac, "D": D,
           "cattle_kept": float((D2[cat_te] < D).mean()), "buffalo_kept": float((D2[buf_te] < D).mean()),
           "other_test_rejected": float((D2[o_te] >= D).mean()), "novel_rejected": float((D2[o_nv] >= D).mean()),
           "novel_by_source": {s: float((D2[o_nv & (source == s)] >= D).mean()) for s in sorted(set(source[o_nv]))}}
    ood_sweep.append(row)
    print(f"  {keep_frac:6.3f} {D:8.1f} {row['cattle_kept']:12.4f} {row['buffalo_kept']:13.4f} "
          f"{row['other_test_rejected']:15.4f} {row['novel_rejected']:10.4f}  "
          + " ".join(f"{s}:{v:.2f}" for s, v in row["novel_by_source"].items()))
note("7: sweep done")
''')

# ------------------------------------------------------------------ 8. evaluate head + combined gate
cell(r'''
def acc(mask):
    return float((pred[mask] == y[mask]).mean()) if mask.sum() else float("nan")
heldout_acc = {n: acc(te & (y == c)) for c, n in enumerate(["cattle", "buffalo", "other"])}
cm = np.zeros((3, 3), int)
for t_, p_ in zip(y[te], pred[te]): cm[t_, p_] += 1
note(f"8: head held-out acc {json.dumps(heldout_acc)}  confusion {cm.tolist()}")

# Softmax P(other) alone -- report its open-set weakness honestly.
print(f"\n  P(other) >= T   in-domain other rejected / NOVEL rejected")
softmax_sweep = []
for T in (0.3, 0.5, 0.7, 0.9):
    r = {"T": T, "cattle_kept": float((P[cat_te, 2] < T).mean()), "buffalo_kept": float((P[buf_te, 2] < T).mean()),
         "other_test_rejected": float((P[o_te, 2] >= T).mean()), "novel_rejected": float((P[o_nv, 2] >= T).mean())}
    softmax_sweep.append(r)
    print(f"  {T:4.2f}  cattle {r['cattle_kept']:.4f} buffalo {r['buffalo_kept']:.4f}  "
          f"other-test {r['other_test_rejected']:.4f}  NOVEL {r['novel_rejected']:.4f}")

# Combined gate: reject if d2 >= D OR p_other >= T.
print(f"\n  COMBINED (d2 >= D or p_other >= T)")
combined_sweep = []
for row in ood_sweep:
    for T in (0.5, 0.7, 0.9):
        rej = (D2 >= row["D"]) | (P[:, 2] >= T)
        c = {"keep_target": row["keep_target"], "D": row["D"], "T": T,
             "cattle_kept": float(~rej[cat_te].mean() + 0) if False else float((~rej[cat_te]).mean()),
             "buffalo_kept": float((~rej[buf_te]).mean()),
             "other_test_rejected": float(rej[o_te].mean()), "novel_rejected": float(rej[o_nv].mean()),
             "novel_by_source": {s: float(rej[o_nv & (source == s)].mean()) for s in sorted(set(source[o_nv]))}}
        combined_sweep.append(c)
        print(f"  keep {c['keep_target']:.3f} D {c['D']:7.1f} T {T:.1f}  cattle {c['cattle_kept']:.4f} "
              f"buffalo {c['buffalo_kept']:.4f}  other-test {c['other_test_rejected']:.4f}  NOVEL {c['novel_rejected']:.4f}")

bov = te & (y < 2)
sp_pred = np.where(P[:, 1] > P[:, 0], 1, 0)
species_report = {
    "overall_acc": float((sp_pred[bov] == y[bov]).mean()),
    "cattle_as_cattle": float((sp_pred[cat_te] == 0).mean()),
    "buffalo_as_buffalo": float((sp_pred[buf_te] == 1).mean()),
    "per_breed": {b: {"n": int((buf_te & (breed == b)).sum()),
                      "acc": float((sp_pred[buf_te & (breed == b)] == 1).mean())} for b in sorted(set(breed[buf_te]))},
}
note(f"8: species {json.dumps(species_report)}")
''')

# ------------------------------------------------------------------ 9. combined model
cell(r'''
species_layer = head.get_layer("species")

# Mahalanobis as two fixed Dense layers + sum of squares. Weights are constants.
proj = keras.layers.Dense(K, use_bias=True, name="ood_project")
whiten = keras.layers.Dense(K, use_bias=False, name="ood_whiten")
z = proj(feat)
w = whiten(z)
d2 = keras.layers.Lambda(lambda t: keras.ops.sum(keras.ops.square(t), axis=-1, keepdims=True), name="ood")(w)
proj.set_weights([P_ood.astype(np.float32), (-(mu @ P_ood)).astype(np.float32)])
whiten.set_weights([L_ood.astype(np.float32)])
proj.trainable = whiten.trainable = False

combined = keras.Model(inp, [lesion, species_layer(feat), d2], name="bi_lsd_v1_gate")

# Lesion did not move; species equals the head; d2 equals numpy.
a = infer.predict(probe, verbose=0).ravel()
outs = combined.predict(probe, verbose=0)
assert np.max(np.abs(a - outs[0].ravel())) < 1e-6
Fp = feature_extractor.predict(probe, verbose=0)
assert np.max(np.abs(head.predict(Fp, verbose=0) - outs[1])) < 1e-5
ref_d2 = d2_of(Fp.astype(np.float64), P_ood, L_ood)
rel = float(np.max(np.abs(outs[2].ravel() - ref_d2) / (ref_d2 + 1e-6)))
assert rel < 1e-3, rel
note(f"9: combined model built; d2 in-graph vs numpy rel err {rel:.2e}")
''')

# ------------------------------------------------------------------ 10. export
cell(r'''
import onnx
from collections import Counter as _C

SM = "/kaggle/working/_savedmodel"
shutil.rmtree(SM, ignore_errors=True)
combined.export(SM)
RAW = "/kaggle/working/_raw.onnx"
r = subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", SM, "--output", RAW,
                    "--opset", "13", "--rename-outputs", "lesion,species,ood"],
                   capture_output=True, text=True)
open("/kaggle/working/tf2onnx.log", "w").write((r.stdout or "") + "\n" + (r.stderr or ""))
assert r.returncode == 0 and os.path.exists(RAW), r.stderr[-800:]

m = onnx.load(RAW)
onnx.checker.check_model(m)
ops = _C(n.op_type for n in m.graph.node)
banned = {k: v for k, v in ops.items() if "Random" in k or "Stateless" in k or k in ("If", "Loop")}
assert not banned, banned
for vi in list(m.graph.input) + list(m.graph.output):
    d0 = vi.type.tensor_type.shape.dim[0]
    if not d0.HasField("dim_value") or d0.dim_value == 0:
        d0.ClearField("dim_param"); d0.dim_value = 1
out_names = [o.name for o in m.graph.output]
assert set(out_names) == {"lesion", "species", "ood"}, out_names

OUT32 = "/kaggle/working/bi-lsd-mnv3l-v1.0.0-gate.onnx"
onnx.save(m, OUT32)
from onnxconverter_common import float16
OUT16 = "/kaggle/working/bi-lsd-mnv3l-v1.0.0-gate.fp16.onnx"
onnx.save(float16.convert_float_to_float16(onnx.load(OUT32), keep_io_types=True), OUT16)
note(f"10: exported  outputs {out_names}  fp32 {os.path.getsize(OUT32)/1e6:.2f} MB  fp16 {os.path.getsize(OUT16)/1e6:.2f} MB")
''')

# ------------------------------------------------------------------ 11. verify ONNX (streaming)
cell(r'''
import onnxruntime as ort
verification = {}
try:
    def session(path):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return s, s.get_inputs()[0].name
    s32, in32 = session(OUT32)
    s16, in16 = session(OUT16)
    def run_one(s, iname, x):
        r = dict(zip([o.name for o in s.get_outputs()], s.run(None, {iname: x[None]})))
        return float(r["lesion"].ravel()[0]), r["species"].ravel().astype(np.float64), float(r["ood"].ravel()[0])

    cattle_test_idx = [i for i, it in enumerate(items) if it["kind"] == "cattle" and it["split"] == "test"]
    yc = np.array([int(r["label"]) for r in cattle_rows if r["split"] == "test"])
    assert len(yc) == len(cattle_test_idx)
    L32, L16, Lk, O32, O16, Ok, S32c, S16c = [], [], [], [], [], [], [], []
    for k, i in enumerate(cattle_test_idx, 1):
        x = load_image(items[i]["ref"])
        a, sa, oa = run_one(s32, in32, x); b, sb, ob = run_one(s16, in16, x)
        L32.append(a); L16.append(b); O32.append(oa); O16.append(ob); S32c.append(sa); S16c.append(sb)
        Lk.append(float(infer.predict(x[None], verbose=0).ravel()[0]))
        Ok.append(float(D2[i]))
        if k % 100 == 0: note(f"11: cattle test {k}/{len(cattle_test_idx)}")
    L32, L16, Lk, O32, O16, Ok = map(np.array, (L32, L16, Lk, O32, O16, Ok))
    S32c, S16c = np.array(S32c), np.array(S16c)

    def metrics(y_, p_):
        yh = (p_ >= 0.5).astype(int)
        tp = int(((yh==1)&(y_==1)).sum()); tn = int(((yh==0)&(y_==0)).sum())
        fp = int(((yh==1)&(y_==0)).sum()); fn = int(((yh==0)&(y_==1)).sum())
        return {"balanced_accuracy": (tp/(tp+fn) + tn/(tn+fp))/2, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    verification["lesion"] = {
        "fp32_vs_keras_max_abs_diff": float(np.max(np.abs(L32 - Lk))),
        "fp16_vs_fp32_max_abs_diff": float(np.max(np.abs(L16 - L32))),
        "fp16_decision_agreement": float(((L16>=0.5)==(L32>=0.5)).mean()),
        "test_metrics_fp32": metrics(yc, L32), "test_metrics_fp16": metrics(yc, L16)}
    verification["ood"] = {
        "fp32_vs_numpy_max_rel_err": float(np.max(np.abs(O32 - Ok) / (Ok + 1e-6))),
        "fp16_vs_fp32_max_rel_err": float(np.max(np.abs(O16 - O32) / (O32 + 1e-6))),
        "fp16_vs_fp32_gate_agreement_at_D": {str(r["keep_target"]): float(((O16 >= r["D"]) == (O32 >= r["D"])).mean()) for r in ood_sweep}}
    verification["species_cattle_test"] = {
        "fp16_vs_fp32_argmax_agreement": float((S16c.argmax(1) == S32c.argmax(1)).mean()),
        "onnx_vs_keras_max_abs_diff": float(np.max(np.abs(S32c - P[cattle_test_idx])))}
    note(f"11: {json.dumps(verification)}")
    np.save("/kaggle/working/test_lesion_probs_fp16.npy", L16)
    np.save("/kaggle/working/test_ood_d2_fp16.npy", O16)

    # novel set through the fp16 artifact -- the number that matters, from the file that ships
    nv_idx = np.where(o_nv)[0]
    rej_counts = {}
    for k, i in enumerate(nv_idx, 1):
        _, sp, od = run_one(s16, in16, load_image(items[i]["ref"]))
        for row in ood_sweep:
            for T in (0.7, 0.8, 0.9):
                key = f"keep{row['keep_target']}_T{T}"
                rej_counts[key] = rej_counts.get(key, 0) + int(od >= row["D"] or sp[2] >= T)
            key = f"keep{row['keep_target']}_oodonly"
            rej_counts[key] = rej_counts.get(key, 0) + int(od >= row["D"])
        if k % 500 == 0: note(f"11: novel {k}/{len(nv_idx)}")
    verification["novel_rejected_fp16_combined"] = {k: v / len(nv_idx) for k, v in rej_counts.items()}
    note(f"11: novel (fp16, combined) {json.dumps(verification['novel_rejected_fp16_combined'])}")
except Exception:
    tb = traceback.format_exc()
    open("/kaggle/working/error.txt", "w").write(tb)
    note("11: FAILED\n" + tb)
''')

# ------------------------------------------------------------------ 12. report
cell(r'''
report = {
    "modelVersion": "bi-lsd-mnv3l-v1.0.0-gate",
    "outputs": {"lesion": [1, 1], "species": {"shape": [1, 3], "classes": ["cattle", "buffalo", "other"]},
                "ood": {"shape": [1, 1], "meaning": f"Mahalanobis d^2 of the 960-d pooled feature from the bovine training distribution, PCA k={K}"}},
    "data": {"counts": {f"{c}/{s}": int(n) for (c, s), n in sorted(Counter(zip(y.tolist(), split.tolist())).items())},
             "novel_sources": dict(Counter(source[o_nv].tolist())),
             "other_train_sources": dict(Counter(source[tr & (y == 2)].tolist()))},
    "lesion_model_on_nonanimals_before_gate": {
        "in_domain_heldout": {"n": int(o_te.sum()), "flagged_rate": float((lesion_p[o_te] >= 0.5).mean())},
        "novel": {"n": int(o_nv.sum()), "flagged_rate": float((lesion_p[o_nv] >= 0.5).mean())}},
    "head": {"heldout_accuracy": heldout_acc, "confusion": cm.tolist(), "species": species_report,
             "softmax_only_sweep": softmax_sweep},
    "ood": {"k": int(K), "k_candidates_auroc": {str(k): cands[k][0] for k in cands}, "sweep": ood_sweep},
    "combined_sweep": combined_sweep,
    "verification": verification,
    "verification_complete": {"lesion", "ood", "novel_rejected_fp16_combined"} <= set(verification),
    "sizes": {"fp32": os.path.getsize(OUT32), "fp16": os.path.getsize(OUT16)},
    "versions": {"tensorflow": tf.__version__, "keras": keras.__version__,
                 "onnx": onnx.__version__, "onnxruntime": ort.__version__},
}
json.dump(report, open("/kaggle/working/gate_report.json", "w"), indent=2)
shutil.rmtree(SM, ignore_errors=True)
if os.path.exists(RAW): os.remove(RAW)
note("12: gate_report.json written; outputs " + str(sorted(os.listdir("/kaggle/working"))))
''')

out = pathlib.Path("ai/kernels/bovine-lsd-gate/bovine_lsd_gate.py")
out.parent.mkdir(parents=True, exist_ok=True)
body = "\n\n".join(f"# {'=' * 74}\n# CELL {i + 1}\n# {'=' * 74}\n{src}" for i, src in enumerate(CELLS))
out.write_text('"""GENERATED by ai/scripts/build_gate_notebook.py -- edit that, not this."""\n\n' + body + "\n",
               encoding="utf-8")
compile(out.read_text(encoding="utf-8"), str(out), "exec")
print(f"wrote {out}  ({len(CELLS)} cells, compiles)")
