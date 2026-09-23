"""Generate the Kaggle kernel for the gate arms study (research plan P1).

One run, four arms, identical data, pre-registered choice:

    arm   l2-normalised distance   farm-background outliers
    A     no                       no        (reproduces shipped v5/v7)
    B     yes                      no        Mahalanobis++ (Mueller & Hein 2025)
    C     no                       yes       outlier exposure (Hendrycks et al. 2019)
    D     yes                      yes

Lesion output is untouched in every arm: same backbone, same Dense layers, and
the kernel asserts byte-level agreement before exporting.

EVALUATION DISCIPLINE -- stricter than the previous run, on purpose:

  * PCA dimension k and both thresholds are chosen on a VALIDATION carve-out
    (10% of train). The locked cattle test split and the novel set are used
    only for reporting and for a pass/fail eligibility check.
  * The novel set (Intel scenes, LFW faces, unseen Caltech classes -- never
    trained on in any arm) is split in half. novel_select chooses the arm;
    novel_report is the number that gets written down. The headline figure is
    therefore not the one used to pick.
  * Eligibility (on the locked test split): >= 99% of cattle, >= 99% of
    buffalo, AND >= 99% of LESION-POSITIVE cattle must pass the gate. The last
    condition exists because of the next point.

LESION-LIKE TEXTURES ARE NEVER TRAINED ON. DTD contains textures -- bumpy,
pitted, studded, scaly, blotchy, wrinkled -- that look like lumpy hide.
Teaching the gate to call those "not an animal" risks rejecting exactly the
close-up skin photographs the app tells farmers to take. They are held out in
every arm as a probe, and diseased-animal retention is a hard eligibility
condition. Only farm / built-environment textures (cracked, grid, stratified,
fibrous, woven, potholed, smeared ...) are added as outliers.

Choice rule, fixed before the run:
  among eligible arms, highest novel_select rejection; within 1 point, the arm
  with fewer changes wins; if nothing beats A by more than 1 point, A stays.
"""
import pathlib

CELLS = []


def cell(src):
    CELLS.append(src.strip("\n"))


# ------------------------------------------------------------------ 1. env
cell(r'''
import sys, os, glob, json, subprocess, shutil, io, csv, zipfile, random, time, traceback, math

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

import numpy as np
import tensorflow as tf
import keras
note(f"1: tf {tf.__version__} keras {keras.__version__} gpus {tf.config.list_physical_devices('GPU')}")
for p in sorted(glob.glob("/kaggle/input/*")):
    print(p)
''')

# ------------------------------------------------------------------ 2. install
cell(r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime",
                "onnxconverter-common"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "tf2onnx"], check=False)
import importlib
for m in ("onnx", "onnxruntime", "tf2onnx", "onnxconverter_common"):
    importlib.import_module(m)
note("2: installs ok")
''')

# ------------------------------------------------------------------ 3. model
cell(r'''
KERAS = sorted(glob.glob("/kaggle/input/**/*.keras", recursive=True))[0]
model = keras.models.load_model(KERAS, compile=False)
backbone = model.get_layer("MobileNetV3Large")
gap      = model.get_layer("global_average_pooling2d")
dense    = model.get_layer("dense")
dense_1  = model.get_layer("dense_1")

inp    = keras.Input(shape=(224, 224, 3), dtype="float32", name="image")
feat   = gap(backbone(inp))
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

def jpgs(root):
    return sorted(p for p in glob.glob(root + "/**/*", recursive=True)
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))

# Kaggle's mount layout changed between runs (the previous run found these at
# /kaggle/input/<slug>; this one did not, and every negative source came back
# EMPTY). So locate each dataset by directory name anywhere under /kaggle/input.
def root_of(slug):
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if os.path.basename(dirpath) == slug:
            return dirpath
        if dirpath.count(os.sep) > 6:
            dirnames[:] = []
    return None
ROOTS = {s: root_of(s) for s in ("natural-images", "rock-paper-scissors", "caltech-101",
                                 "intel-image-classification", "lfw-dataset",
                                 "describable-textures-dataset-dtd")}
note("4: dataset roots " + json.dumps(ROOTS))
missing = [s for s, r in ROOTS.items() if r is None]
if missing:
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if dirpath.count(os.sep) <= 5: note("   tree: " + dirpath)
    raise SystemExit(f"dataset roots not found: {missing}")

nat = sorted({"/".join(p.split("/")[-2:]): p for p in jpgs(ROOTS["natural-images"])}.values())
hands   = jpgs(ROOTS["rock-paper-scissors"])
caltech = jpgs(ROOTS["caltech-101"])
scenes  = jpgs(ROOTS["intel-image-classification"])
faces   = jpgs(ROOTS["lfw-dataset"])
dtd     = [p for p in jpgs(ROOTS["describable-textures-dataset-dtd"]) if "/images/" in p]
dtd     = sorted({"/".join(p.split("/")[-2:]): p for p in dtd}.values())

rnd = random.Random(42)
rnd.shuffle(hands);  hands  = hands[:1500]
rnd.shuffle(scenes); scenes = scenes[:3000]
rnd.shuffle(faces);  faces  = faces[:1000]

cal_by_class = {}
for p in caltech:
    cal_by_class.setdefault(p.split("/")[-2], []).append(p)
novel_classes = set(sorted(cal_by_class)[2::3])     # identical to the previous run

# DTD: farm / built-environment textures are candidate outliers (arms C, D);
# lesion-like textures are NEVER trained on, in any arm.
FARM = {"banded", "braided", "chequered", "cracked", "crosshatched", "fibrous", "grid",
        "grooved", "interlaced", "lined", "matted", "meshed", "perforated", "potholed",
        "smeared", "stained", "stratified", "striped", "woven", "zigzagged", "knitted",
        "lacelike", "marbled", "paisley", "gauzy", "cobwebbed", "spiralled", "swirly", "waffled"}
LESIONLIKE = {"bumpy", "blotchy", "bubbly", "crystalline", "dotted", "flecked", "freckled",
              "frilly", "honeycombed", "pitted", "pleated", "polka-dotted", "porous", "scaly",
              "sprinkled", "studded", "veined", "wrinkled"}
dtd_by_class = {}
for p in dtd:
    dtd_by_class.setdefault(p.split("/")[-2], []).append(p)
unknown = set(dtd_by_class) - FARM - LESIONLIKE
if unknown:
    note(f"4: WARNING unrecognised DTD classes treated as lesion-like (never trained): {sorted(unknown)}")
    LESIONLIKE |= unknown
note(f"4: cattle {len(cattle_rows)} buffalo {len(buf_unique)} natural {len(nat)} hands {len(hands)} "
     f"caltech {len(caltech)} scenes {len(scenes)} faces {len(faces)} dtd {len(dtd)} "
     f"(farm classes {len(FARM & set(dtd_by_class))}, lesion-like {len(LESIONLIKE & set(dtd_by_class))})")

# ---- items. cls 0 cattle, 1 buffalo, 2 other ---------------------------------
items = []
def add(ref, cls, split, source, **kw):
    items.append(dict(ref=ref, cls=cls, split=split, source=source, extra=kw.pop("extra", False),
                      label=kw.pop("label", -1), severity=kw.pop("severity", ""), breed=kw.pop("breed", "")))

for r in cattle_rows:
    add(os.path.join(cattle_root, r["file"]), 0, "test" if r["split"] == "test" else "train",
        "cattle_" + r["source"], label=int(r["label"]), severity=r.get("severity", ""))
by_breed = {}
for r in buf_unique:
    by_breed.setdefault(r["breed"], []).append(r)
for breed, rows in sorted(by_breed.items()):
    rnd.shuffle(rows)
    n_test = max(1, round(0.25 * len(rows)))
    for i, r in enumerate(rows):
        add(os.path.join(buf_root, r["file"]), 1, "test" if i < n_test else "train", "buffalo", breed=breed)

def split80(_):
    return "test" if rnd.random() < 0.2 else "train"
for p in nat:   add(p, 2, split80(p), "natural")
for p in hands: add(p, 2, split80(p), "hands")
for c, paths in sorted(cal_by_class.items()):
    for p in paths:
        add(p, 2, "novel" if c in novel_classes else split80(p), "caltech_novel" if c in novel_classes else "caltech")
for p in scenes: add(p, 2, "novel", "scenes")
for p in faces:  add(p, 2, "novel", "faces")
for c, paths in sorted(dtd_by_class.items()):
    for p in paths:
        if c in FARM:
            add(p, 2, split80(p), "dtd_farm", extra=True)
        else:
            add(p, 2, "probe", "dtd_lesionlike")

# ---- carve-outs: 10% of train -> val; novel -> select / report halves ---------
vr, nr = random.Random(7), random.Random(11)
for it in items:
    if it["split"] == "train" and vr.random() < 0.10:
        it["split"] = "val"
    elif it["split"] == "novel":
        it["split"] = "novel_select" if nr.random() < 0.5 else "novel_report"

# Refuse to continue on empty data. This run's first attempt computed four
# arms on zero negatives and only failed at the very end.
counts = {k: len(v) for k, v in dict(natural=nat, hands=hands, caltech=caltech, scenes=scenes,
                                     faces=faces, dtd=dtd).items()}
assert all(n > 0 for n in counts.values()), f"empty negative source(s): {counts}"
note("4: splits " + json.dumps({f"{c}/{s}": n for (c, s), n in
     sorted(Counter((it["cls"], it["split"]) for it in items).items())}))

def load_image(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32)
''')

# ------------------------------------------------------------------ 5. features (once; shared by all arms)
cell(r'''
t0 = time.time()
BATCH = 64
feats = np.zeros((len(items), 960), np.float32)
lesion_p = np.zeros(len(items), np.float32)
bad = 0
for s in range(0, len(items), BATCH):
    chunk = items[s:s + BATCH]
    imgs = []
    for it in chunk:
        try:
            imgs.append(load_image(it["ref"]))
        except Exception:
            imgs.append(np.zeros((224, 224, 3), np.float32)); it["bad"] = True; bad += 1
    batch = np.stack(imgs)
    feats[s:s + len(batch)] = feature_extractor.predict(batch, verbose=0)
    lesion_p[s:s + len(batch)] = infer.predict(batch, verbose=0).ravel()
    if (s // BATCH) % 50 == 0:
        note(f"5: {s + len(batch)}/{len(items)}  {time.time() - t0:.0f}s")
keep = np.array([not it.get("bad") for it in items])
items = [it for it, k in zip(items, keep) if k]
feats, lesion_p = feats[keep], lesion_p[keep]
note(f"5: features {feats.shape} in {time.time() - t0:.0f}s (unreadable dropped: {bad})")

y      = np.array([it["cls"] for it in items])
split  = np.array([it["split"] for it in items])
source = np.array([it["source"] for it in items])
extra  = np.array([it["extra"] for it in items])
label  = np.array([it["label"] for it in items])
sev    = np.array([it["severity"] for it in items])

# Mahalanobis++: l2-normalise, then rescale to sqrt(960) so values stay in a
# comfortable fp16 range after export. Mahalanobis distance is invariant to
# that constant (covariance and shrinkage scale with it).
SCALE = float(math.sqrt(960))
feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12) * SCALE

M = {
    "cattle_test":   (split == "test") & (y == 0),
    "buffalo_test":  (split == "test") & (y == 1),
    "lesion_test":   (split == "test") & (y == 0) & (label == 1),
    "mild_test":     (split == "test") & (y == 0) & (label == 1) & (sev == "Mild"),
    "cattle_val":    (split == "val") & (y == 0),
    "buffalo_val":   (split == "val") & (y == 1),
    "lesion_val":    (split == "val") & (y == 0) & (label == 1),
    "other_test":    (split == "test") & (y == 2) & ~extra,     # same set as the previous run
    "dtd_farm_test": (split == "test") & (source == "dtd_farm"),
    "lesionlike":    (split == "probe"),
    "novel_select":  (split == "novel_select"),
    "novel_report":  (split == "novel_report"),
}
note("5: masks " + json.dumps({k: int(v.sum()) for k, v in M.items()}))
assert all(int(v.sum()) > 0 for v in M.values()), {k: int(v.sum()) for k, v in M.items()}
note(f"5: SHIPPED lesion model flags as 'possible condition': novel_report "
     f"{float((lesion_p[M['novel_report']] >= 0.5).mean()):.1%}, lesion-like textures "
     f"{float((lesion_p[M['lesionlike']] >= 0.5).mean()):.1%}")
''')

# ------------------------------------------------------------------ 6. arm machinery
cell(r'''
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def fit_head(include_extra):
    trm = (split == "train") & (include_extra | ~extra)
    vam = (split == "val") & (include_extra | ~extra)
    counts = np.bincount(y[trm], minlength=3)
    cw = {c: float(trm.sum() / (3 * counts[c])) for c in range(3)}
    keras.utils.set_random_seed(42)
    head = keras.Sequential([keras.Input(shape=(960,)),
                             keras.layers.Dense(3, activation="softmax", name="species",
                                                kernel_regularizer=keras.regularizers.l2(1e-4))])
    head.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    head.fit(feats[trm], y[trm], validation_data=(feats[vam], y[vam]), epochs=80, batch_size=128,
             class_weight=cw, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return head, head.predict(feats, verbose=0)

def fit_ood(normalise):
    F = feats_n if normalise else feats
    Xb = F[(split == "train") & (y < 2)]
    mu = Xb.mean(0)
    _, _, Vt = np.linalg.svd(Xb - mu, full_matrices=False)
    def build(k, shrink=0.05):
        Pk = Vt[:k].T.astype(np.float64)
        Z = (Xb - mu) @ Pk
        cov = np.cov(Z, rowvar=False)
        cov = (1 - shrink) * cov + shrink * np.trace(cov) / k * np.eye(k)
        L = np.linalg.cholesky(np.linalg.inv(cov))
        W = ((F - mu) @ Pk) @ L
        return Pk, L, np.einsum("ij,ij->i", W, W)
    # k chosen on VALIDATION only: bovine-val vs other-val (never test, never novel)
    bov_val = (split == "val") & (y < 2)
    oth_val = (split == "val") & (y == 2) & ~extra
    best = None
    for k in (128, 256, 384, 512):
        Pk, L, d2 = build(k)
        a = auroc(d2[oth_val], d2[bov_val])
        note(f"   ood normalise={normalise} k={k} AUROC(val)={a:.4f}")
        if best is None or a > best[0]:
            best = (a, k, Pk, L, d2)
    a, k, Pk, L, d2 = best
    return {"k": k, "auroc_val": a, "P": Pk, "L": L, "mu": mu, "d2": d2, "normalise": normalise}

def choose_thresholds(P, d2, include_extra):
    """Grid on VALIDATION: keep >= 99.5% of cattle-val, lesion-val and 100% of
    buffalo-val (it is ~24 images), maximise other-val rejection. Ties go to the
    more permissive setting."""
    oth_val = (split == "val") & (y == 2) & (include_extra | ~extra)
    bov_val = (split == "val") & (y < 2)
    Ds = sorted(set(float(np.percentile(d2[bov_val], q)) for q in (99.0, 99.5, 99.8, 100.0)) | {float(d2[bov_val].max() * 1.25)})
    Ts = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    best = None
    for D in Ds:
        for T in Ts:
            rej = (d2 >= D) | (P[:, 2] >= T)
            kc = float((~rej[M["cattle_val"]]).mean())
            kl = float((~rej[M["lesion_val"]]).mean())
            kb = float((~rej[M["buffalo_val"]]).mean())
            if kc < 0.995 or kl < 0.995 or kb < 1.0:
                continue
            score = (float(rej[oth_val].mean()), D, T)      # tie -> larger D, larger T
            if best is None or score > best[0]:
                best = (score, D, T)
    if best is None:                                         # nothing qualifies: most permissive
        return max(Ds), 0.95, float("nan")
    return best[1], best[2], best[0][0]

def evaluate(P, d2, D, T):
    rej = (d2 >= D) | (P[:, 2] >= T)
    r = lambda m: float(rej[m].mean())
    k = lambda m: float((~rej[m]).mean())
    by_src = {s: float(rej[M["novel_report"] & (source == s)].mean())
              for s in sorted(set(source[M["novel_report"]]))}
    sp = np.where(P[:, 1] > P[:, 0], 1, 0)
    return {
        "D": D, "T": T,
        "retention": {"cattle_test": k(M["cattle_test"]), "buffalo_test": k(M["buffalo_test"]),
                      "lesion_positive_test": k(M["lesion_test"]), "mild_test": k(M["mild_test"])},
        "rejection": {"novel_select": r(M["novel_select"]), "novel_report": r(M["novel_report"]),
                      "novel_report_by_source": by_src, "other_test": r(M["other_test"]),
                      "dtd_farm_test": r(M["dtd_farm_test"]), "lesionlike_probe": r(M["lesionlike"])},
        "ood_only_novel_report": float((d2[M["novel_report"]] >= D).mean()),
        "softmax_only_novel_report": float((P[M["novel_report"], 2] >= T).mean()),
        "species": {"cattle_as_cattle": float((sp[M["cattle_test"]] == 0).mean()),
                    "buffalo_as_buffalo": float((sp[M["buffalo_test"]] == 1).mean())},
    }
note("6: machinery ready")
''')

# ------------------------------------------------------------------ 7. run arms
cell(r'''
ARMS = {"A": dict(normalise=False, extra=False), "B": dict(normalise=True, extra=False),
        "C": dict(normalise=False, extra=True),  "D": dict(normalise=True, extra=True)}
heads = {False: fit_head(False), True: fit_head(True)}
note("7: heads trained")
oods = {False: fit_ood(False), True: fit_ood(True)}
note("7: ood fits done")

results, artefacts = {}, {}
for name, cfg in ARMS.items():
    head, P = heads[cfg["extra"]]
    ood = oods[cfg["normalise"]]
    D, T, val_rej = choose_thresholds(P, ood["d2"], cfg["extra"])
    ev = evaluate(P, ood["d2"], D, T)
    ev.update({"config": cfg, "k": ood["k"], "ood_auroc_val": ood["auroc_val"], "other_val_rejection": val_rej})
    ev["eligible"] = (ev["retention"]["cattle_test"] >= 0.99 and ev["retention"]["buffalo_test"] >= 0.99
                      and ev["retention"]["lesion_positive_test"] >= 0.99)
    results[name] = ev
    artefacts[name] = (head, ood, D, T)
    note(f"7: arm {name}  D={D:.1f} T={T}  keep cattle {ev['retention']['cattle_test']:.4f} "
         f"buffalo {ev['retention']['buffalo_test']:.4f} lesion+ {ev['retention']['lesion_positive_test']:.4f} "
         f"mild {ev['retention']['mild_test']:.4f} | reject novel_select {ev['rejection']['novel_select']:.4f} "
         f"novel_report {ev['rejection']['novel_report']:.4f} lesionlike {ev['rejection']['lesionlike_probe']:.4f} "
         f"| eligible {ev['eligible']}")

# ---- pre-registered choice -------------------------------------------------
changes = {"A": 0, "B": 1, "C": 1, "D": 2}
eligible = [a for a in ARMS if results[a]["eligible"]]
base = results["A"]["rejection"]["novel_select"]
choice, reason = "A", "default"
if eligible:
    top = max(results[a]["rejection"]["novel_select"] for a in eligible)
    within = [a for a in eligible if top - results[a]["rejection"]["novel_select"] <= 0.01]
    cand = min(within, key=lambda a: (changes[a], -results[a]["rejection"]["novel_select"]))
    if cand != "A" and results[cand]["rejection"]["novel_select"] - base > 0.01:
        choice, reason = cand, "beats A by >1 point on novel_select; fewest changes within 1 point of best"
    else:
        choice, reason = "A", "no eligible arm beats A by more than 1 point on novel_select"
if "A" not in eligible and choice == "A":
    reason += " (NOTE: A itself fails eligibility under this run's stricter val-chosen thresholds)"
note(f"7: CHOICE = {choice}  ({reason})")
''')

# ------------------------------------------------------------------ 8. build + verify chosen model
cell(r'''
head, ood, D_ch, T_ch = artefacts[choice]
K = ood["k"]
ood_in = feat
if ood["normalise"]:
    ood_in = keras.layers.Lambda(
        lambda t: t * (SCALE / keras.ops.sqrt(keras.ops.sum(keras.ops.square(t), axis=-1, keepdims=True) + 1e-12)),
        name="ood_l2norm")(feat)
proj   = keras.layers.Dense(K, use_bias=True, name="ood_project")
whiten = keras.layers.Dense(K, use_bias=False, name="ood_whiten")
w  = whiten(proj(ood_in))
d2 = keras.layers.Lambda(lambda t: keras.ops.sum(keras.ops.square(t), axis=-1, keepdims=True), name="ood")(w)
proj.set_weights([ood["P"].astype(np.float32), (-(ood["mu"] @ ood["P"])).astype(np.float32)])
whiten.set_weights([ood["L"].astype(np.float32)])
species_layer = keras.layers.Dense(3, activation="softmax", name="species_out")
_ = species_layer(feat)
species_layer.set_weights(head.get_layer("species").get_weights())
combined = keras.Model(inp, [lesion, species_layer(feat), d2], name=f"bi_lsd_gate_arm{choice}")

outs = combined.predict(probe, verbose=0)
assert np.max(np.abs(infer.predict(probe, verbose=0).ravel() - outs[0].ravel())) < 1e-6
Fp = feature_extractor.predict(probe, verbose=0).astype(np.float64)
assert np.max(np.abs(head.predict(Fp, verbose=0) - outs[1])) < 1e-5
Fq = Fp / (np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12) * SCALE if ood["normalise"] else Fp
Wq = ((Fq - ood["mu"]) @ ood["P"]) @ ood["L"]
ref = np.einsum("ij,ij->i", Wq, Wq)
rel = float(np.max(np.abs(outs[2].ravel() - ref) / (ref + 1e-6)))
assert rel < 1e-3, rel
note(f"8: combined arm {choice} built; d2 in-graph vs numpy rel err {rel:.2e}")

import onnx
from collections import Counter as _C
SM = "/kaggle/working/_savedmodel"; shutil.rmtree(SM, ignore_errors=True)
combined.export(SM)
RAW = "/kaggle/working/_raw.onnx"
r = subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", SM, "--output", RAW,
                    "--opset", "13", "--rename-outputs", "lesion,species,ood"], capture_output=True, text=True)
open("/kaggle/working/tf2onnx.log", "w").write((r.stdout or "") + "\n" + (r.stderr or ""))
assert r.returncode == 0 and os.path.exists(RAW), r.stderr[-800:]
m = onnx.load(RAW); onnx.checker.check_model(m)
ops = _C(n.op_type for n in m.graph.node)
assert not {k for k in ops if "Random" in k or "Stateless" in k or k in ("If", "Loop")}, ops
for vi in list(m.graph.input) + list(m.graph.output):
    d0 = vi.type.tensor_type.shape.dim[0]
    if not d0.HasField("dim_value") or d0.dim_value == 0:
        d0.ClearField("dim_param"); d0.dim_value = 1
assert {o.name for o in m.graph.output} == {"lesion", "species", "ood"}
OUT32 = f"/kaggle/working/bi-lsd-mnv3l-v1.0.0-gate-{choice}.onnx"
onnx.save(m, OUT32)
from onnxconverter_common import float16
OUT16 = f"/kaggle/working/bi-lsd-mnv3l-v1.0.0-gate-{choice}.fp16.onnx"
onnx.save(float16.convert_float_to_float16(onnx.load(OUT32), keep_io_types=True), OUT16)
note(f"8: exported fp32 {os.path.getsize(OUT32)/1e6:.2f} MB, fp16 {os.path.getsize(OUT16)/1e6:.2f} MB")
''')

# ------------------------------------------------------------------ 9. verify through the exported file
cell(r'''
import onnxruntime as ort
verification = {}
try:
    def sess(p):
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        return s, s.get_inputs()[0].name, [o.name for o in s.get_outputs()]
    s32, i32, n32 = sess(OUT32); s16, i16, n16 = sess(OUT16)
    def run(s, i, n, x):
        o = dict(zip(n, s.run(None, {i: x[None]})))
        return float(o["lesion"].ravel()[0]), o["species"].ravel().astype(np.float64), float(o["ood"].ravel()[0])

    idx_ct = np.where(M["cattle_test"])[0]
    yc = label[idx_ct]
    L32, L16, O32, O16, S16 = [], [], [], [], []
    for n, i in enumerate(idx_ct, 1):
        x = load_image(items[i]["ref"])
        a, _, oa = run(s32, i32, n32, x); b, sb, ob = run(s16, i16, n16, x)
        L32.append(a); L16.append(b); O32.append(oa); O16.append(ob); S16.append(sb)
        if n % 150 == 0: note(f"9: cattle test {n}/{len(idx_ct)}")
    L32, L16, O32, O16, S16 = map(np.array, (L32, L16, O32, O16, S16))
    def cm(y_, p_):
        yh = (p_ >= 0.5).astype(int)
        return {"tp": int(((yh == 1) & (y_ == 1)).sum()), "tn": int(((yh == 0) & (y_ == 0)).sum()),
                "fp": int(((yh == 1) & (y_ == 0)).sum()), "fn": int(((yh == 0) & (y_ == 1)).sum())}
    verification["lesion_fp32_vs_keras"] = float(np.max(np.abs(L32 - lesion_p[idx_ct])))
    verification["lesion_confusion_fp32"] = cm(yc, L32)
    verification["lesion_confusion_fp16"] = cm(yc, L16)
    verification["ood_fp32_vs_numpy_max_rel"] = float(np.max(np.abs(O32 - ood["d2"][idx_ct]) / (ood["d2"][idx_ct] + 1e-6)))
    verification["ood_fp16_vs_fp32_max_rel"] = float(np.max(np.abs(O16 - O32) / (O32 + 1e-6)))
    gate16 = (O16 >= D_ch) | (S16[:, 2] >= T_ch)
    verification["cattle_test_kept_fp16"] = float((~gate16).mean())
    note(f"9: {json.dumps(verification)}")

    idx_nr = np.where(M["novel_report"])[0]
    rej = 0
    for n, i in enumerate(idx_nr, 1):
        _, sp, od = run(s16, i16, n16, load_image(items[i]["ref"]))
        rej += int(od >= D_ch or sp[2] >= T_ch)
        if n % 500 == 0: note(f"9: novel_report {n}/{len(idx_nr)}")
    verification["novel_report_rejected_fp16"] = rej / len(idx_nr)
    note(f"9: novel_report rejected through the fp16 file: {verification['novel_report_rejected_fp16']:.4f}")
except Exception:
    open("/kaggle/working/error.txt", "w").write(traceback.format_exc())
    note("9: FAILED\n" + traceback.format_exc())
''')

# ------------------------------------------------------------------ 10. report
cell(r'''
report = {
    "study": "gate arms A/B/C/D (research plan P1)",
    "choice": choice, "choice_reason": reason,
    "rule": "eligible = test retention >= 0.99 for cattle, buffalo and lesion-positive cattle; "
            "choose highest novel_select rejection; within 1 point, fewest changes; must beat A by >1 point",
    "thresholds_chosen_on": "validation carve-out (10% of train); k chosen on validation AUROC",
    "arms": results,
    "chosen_thresholds": {"oodMax": D_ch, "otherMax": T_ch, "k": K, "normalise": ood["normalise"]},
    "verification": verification,
    "sizes": {"fp32": os.path.getsize(OUT32), "fp16": os.path.getsize(OUT16)},
    "masks": {k: int(v.sum()) for k, v in M.items()},
    "dtd_farm_classes": sorted(FARM & set(dtd_by_class)),
    "dtd_lesionlike_classes_never_trained": sorted(LESIONLIKE & set(dtd_by_class)),
}
json.dump(report, open("/kaggle/working/arms_report.json", "w"), indent=2, default=float)
shutil.rmtree(SM, ignore_errors=True)
if os.path.exists(RAW): os.remove(RAW)
note("10: arms_report.json written; outputs " + str(sorted(os.listdir("/kaggle/working"))))
''')

out = pathlib.Path("ai/kernels/bovine-lsd-gate-arms/bovine_lsd_gate_arms.py")
out.parent.mkdir(parents=True, exist_ok=True)
body = "\n\n".join(f"# {'=' * 74}\n# CELL {i + 1}\n# {'=' * 74}\n{src}" for i, src in enumerate(CELLS))
out.write_text('"""GENERATED by ai/scripts/build_gate_arms_notebook.py -- edit that, not this."""\n\n' + body + "\n",
               encoding="utf-8")
compile(out.read_text(encoding="utf-8"), str(out), "exec")
print(f"wrote {out}  ({len(CELLS)} cells, compiles)")
