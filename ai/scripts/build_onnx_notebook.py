"""Generate the Kaggle notebook that converts bi-lsd-mnv3l-v1.0.0.keras -> ONNX.

We have no TensorFlow locally (no GPU, Windows). Kaggle has TF preinstalled.
The kernel converts, then verifies Keras <-> ONNX <-> TFLite agreement on the
LOCKED TEST SPLIT before we trust the artifact. Nothing gets built on top of an
unverified ONNX file -- a preprocessing/graph mismatch is exactly what killed
the predecessor model (MODEL_CARD.md section 7).
"""
import json
import pathlib

CELLS = []


def cell(src):
    CELLS.append(src.strip("\n"))


# ------------------------------------------------------------------ 1. env
cell(r'''
import sys, os, glob, json, subprocess, shutil
print("python     ", sys.version.split()[0])
import numpy as np;      print("numpy      ", np.__version__)
import tensorflow as tf; print("tensorflow ", tf.__version__)
import keras;            print("keras      ", keras.__version__)

print("\n--- /kaggle/input ---")
for p in sorted(glob.glob("/kaggle/input/*")):
    print(p)
    for q in sorted(glob.glob(p + "/*"))[:12]:
        print("   ", q, (os.path.getsize(q) if os.path.isfile(q) else "<dir>"))
''')

# ------------------------------------------------------------------ 2. install
cell(r'''
# tf2onnx pins protobuf~=3.20, which would break the preinstalled TensorFlow.
# --no-deps installs the converter without dragging that pin in; its real
# runtime deps (numpy, onnx, flatbuffers, six, requests) are already present.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "tf2onnx"], check=False)

import importlib
for m in ("onnx", "onnxruntime", "tf2onnx", "tensorflow"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:12} IMPORT FAILED: {type(e).__name__}: {e}")
''')

# ------------------------------------------------------------------ 3. load keras
cell(r'''
KERAS  = sorted(glob.glob("/kaggle/input/**/*.keras",  recursive=True))[0]
TFLITE = sorted(glob.glob("/kaggle/input/**/*.tflite", recursive=True))[0]
print("keras :", KERAS,  os.path.getsize(KERAS))
print("tflite:", TFLITE, os.path.getsize(TFLITE))

model = keras.models.load_model(KERAS, compile=False)
print("\ninput :", model.inputs[0].shape, model.inputs[0].dtype)
print("output:", model.outputs[0].shape, model.outputs[0].dtype)
print("params:", model.count_params())

# Confirm the rescaling is INSIDE the graph -- the whole input contract rests on it.
print("\n--- first layers ---")
budget = [14]
def walk(layer, depth=0):
    if budget[0] <= 0:
        return
    budget[0] -= 1
    print("  " * depth + f"{layer.__class__.__name__:22} {getattr(layer, 'name', '')}")
    if layer.__class__.__name__ == "Rescaling":
        print("  " * depth + f"   -> scale={layer.scale} offset={layer.offset}")
    for sub in getattr(layer, "layers", []):
        walk(sub, depth + 1)
for l in model.layers:
    walk(l)

print("\n--- last layer ---")
last = model.layers[-1]
print(last.__class__.__name__, getattr(last, "activation", None))
''')

# ------------------------------------------------------------------ 3b. strip augmentation
cell(r'''
# The saved graph is:
#     input -> Sequential "augment" -> Functional "MobileNetV3Large" -> head
#
# "augment" is RandomFlip/Rotation/Zoom/Brightness/Contrast. Those are no-ops at
# inference, but model.export() TRACES them, and the traced graph keeps five
# StatelessRandomUniformV2 nodes. ONNX has no such operator, so the first export
# attempt produced a graph that failed onnx.checker outright -- and had it
# loaded, it would have randomly flipped and recoloured every photo at
# inference time.
#
# So export an inference-only model instead: the backbone (which CONTAINS the
# Rescaling, so the raw [0,255] contract is preserved) plus the head, with the
# augmentation and the two Dropouts -- both identities at inference -- dropped.
# Cell 3c proves this rebuild is equivalent before anything is exported.

backbone = model.get_layer("MobileNetV3Large")
print("backbone first layer:", backbone.layers[1].__class__.__name__,
      "scale=", getattr(backbone.layers[1], "scale", None),
      "offset=", getattr(backbone.layers[1], "offset", None))

inp = keras.Input(shape=(224, 224, 3), dtype="float32", name="image")
h = backbone(inp)
h = model.get_layer("global_average_pooling2d")(h)
h = model.get_layer("dense")(h)       # 128, relu
out = model.get_layer("dense_1")(h)   # 1, sigmoid
infer = keras.Model(inp, out, name="bi_lsd_infer")

print()
infer.summary(line_length=100)
print("\nparams original:", model.count_params(), " inference:", infer.count_params())
''')

# ------------------------------------------------------------------ 3c. equivalence
cell(r'''
# Prove the rebuilt model IS the original at inference before exporting it.
rng = np.random.default_rng(42)
probe = rng.uniform(0, 255, size=(8, 224, 224, 3)).astype(np.float32)

a = model.predict(probe, verbose=0).ravel()
b = infer.predict(probe, verbose=0).ravel()
d = float(np.max(np.abs(a - b)))
print("original :", np.array2string(a, precision=6))
print("inference:", np.array2string(b, precision=6))
print("max abs diff:", d)
assert d < 1e-5, f"rebuilt inference model diverges from the original: {d}"
print("OK: augmentation/dropout removal is inference-equivalent")
''')

# ------------------------------------------------------------------ 4. savedmodel
cell(r'''
SM = "/kaggle/working/_savedmodel"
shutil.rmtree(SM, ignore_errors=True)
infer.export(SM)
print(sorted(os.listdir(SM)))

loaded = tf.saved_model.load(SM)
sig = loaded.signatures["serving_default"]
print("\nsignature inputs :", sig.structured_input_signature)
print("signature outputs:", sig.structured_outputs)
''')

# ------------------------------------------------------------------ 5. convert
cell(r'''
OPSET = 13
ONNX_RAW = "/kaggle/working/_raw.onnx"
attempts = []

def try_convert(name, argv):
    print("=" * 70); print("STRATEGY:", name); print("=" * 70)
    if os.path.exists(ONNX_RAW):
        os.remove(ONNX_RAW)
    r = subprocess.run(argv, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(ONNX_RAW)
    print("returncode:", r.returncode, " produced file:", os.path.exists(ONNX_RAW))
    print((r.stdout or "")[-2500:])
    print((r.stderr or "")[-5000:])
    attempts.append({"strategy": name, "returncode": r.returncode, "ok": bool(ok)})
    return ok

ok = try_convert("A: tf2onnx --saved-model", [
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", SM, "--output", ONNX_RAW, "--opset", str(OPSET),
])

if not ok:
    ok = try_convert("B: tf2onnx --tflite", [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", TFLITE, "--output", ONNX_RAW, "--opset", str(OPSET),
    ])

assert ok, f"all conversion strategies failed: {attempts}"
print("\nraw onnx size MB:", round(os.path.getsize(ONNX_RAW) / 1e6, 2))
''')

# ------------------------------------------------------------------ 6. pin batch
cell(r'''
import onnx
from collections import Counter

m_onnx = onnx.load(ONNX_RAW)
onnx.checker.check_model(m_onnx)

def dims(vi):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in vi.type.tensor_type.shape.dim]

print("ir_version:", m_onnx.ir_version)
print("opsets    :", [(o.domain or "ai.onnx", o.version) for o in m_onnx.opset_import])
for vi in m_onnx.graph.input:
    print("IN  ", vi.name, dims(vi), onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type))
for vi in m_onnx.graph.output:
    print("OUT ", vi.name, dims(vi), onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type))
ops = Counter(n.op_type for n in m_onnx.graph.node)
print("\nop histogram:", dict(ops))

# Hard gate. Version 1 of this export shipped five StatelessRandomUniformV2
# nodes because model.export() traced the training-time augmentation layers.
# Never again, silently.
banned = {k: v for k, v in ops.items()
          if "Random" in k or "Stateless" in k or k in ("If", "Loop")}
assert not banned, f"training-time / control-flow ops leaked into the graph: {banned}"
print("gate: no random or control-flow ops in the graph")

# Pin batch to 1. The browser only ever runs one image, and a static batch lets
# onnxruntime-web fold shapes at load time instead of on every run.
for vi in list(m_onnx.graph.input) + list(m_onnx.graph.output):
    d0 = vi.type.tensor_type.shape.dim[0]
    if not d0.HasField("dim_value") or d0.dim_value == 0:
        d0.ClearField("dim_param")
        d0.dim_value = 1

ONNX_OUT = "/kaggle/working/bi-lsd-mnv3l-v1.0.0.onnx"
onnx.save(m_onnx, ONNX_OUT)
onnx.checker.check_model(onnx.load(ONNX_OUT))

INPUT_NAME  = m_onnx.graph.input[0].name
OUTPUT_NAME = m_onnx.graph.output[0].name
print("\nshipping artifact:", ONNX_OUT, round(os.path.getsize(ONNX_OUT) / 1e6, 2), "MB")
print("input name :", INPUT_NAME,  dims(m_onnx.graph.input[0]))
print("output name:", OUTPUT_NAME, dims(m_onnx.graph.output[0]))
''')

# ------------------------------------------------------------------ 7. test images
cell(r'''
import csv, io, zipfile
from PIL import Image

manifests = glob.glob("/kaggle/input/**/manifest.csv", recursive=True)
zips = [z for z in glob.glob("/kaggle/input/**/*.zip", recursive=True) if "bundle" in z.lower()]
print("manifests:", manifests)
print("zips     :", zips)

if manifests:
    MANIFEST = manifests[0]
    ROOT = os.path.dirname(MANIFEST)
    rows = list(csv.DictReader(open(MANIFEST)))
    def read_image(rel):
        return Image.open(os.path.join(ROOT, rel))
elif zips:
    ZF = zipfile.ZipFile(zips[0])
    rows = list(csv.DictReader(io.StringIO(ZF.read("manifest.csv").decode())))
    def read_image(rel):
        return Image.open(io.BytesIO(ZF.read(rel)))
else:
    raise SystemExit("no manifest.csv and no bundle zip under /kaggle/input")

test = [r for r in rows if r["split"] == "test"]
print(f"\nmanifest rows {len(rows)}  test rows {len(test)}")
print("sample:", test[0])

def preprocess(rel):
    """EXACTLY mirrors ai/scripts/verify_v1.py: RGB, bilinear 224x224, RAW [0,255]."""
    im = read_image(rel).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32)[None, ...]
''')

# ------------------------------------------------------------------ 8. parity
cell(r'''
import onnxruntime as ort

sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
print("ort inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("ort outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
idet, odet = interp.get_input_details()[0], interp.get_output_details()[0]

N = len(test)
p_onnx, p_tfl, p_keras, y, sev, src = [], [], [], [], [], []

for k, r in enumerate(test, 1):
    x = preprocess(r["file"])
    p_onnx.append(float(sess.run(None, {INPUT_NAME: x})[0].ravel()[0]))
    interp.set_tensor(idet["index"], x)
    interp.invoke()
    p_tfl.append(float(interp.get_tensor(odet["index"]).ravel()[0]))
    p_keras.append(float(infer.predict(x, verbose=0).ravel()[0]))
    y.append(int(r["label"]))
    sev.append(r.get("severity", ""))
    src.append(r.get("source", ""))
    if k % 50 == 0 or k == N:
        print(f"\r  {k}/{N}", end="", flush=True)
print()

p_onnx  = np.array(p_onnx)
p_tfl   = np.array(p_tfl)
p_keras = np.array(p_keras)
y, sev, src = np.array(y), np.array(sev), np.array(src)
''')

# ------------------------------------------------------------------ 9. report
cell(r'''
TAU = 0.50

def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p), float); s = p[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def metrics(p):
    yh = (p >= TAU).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    rec, spec = tp / (tp + fn), tn / (tn + fp)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (rec + spec) / 2,
            "auc": float(auc(y, p)), "recall": rec, "specificity": spec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def agree(a, b):
    return float(((a >= TAU) == (b >= TAU)).mean())

report = {
    "modelVersion": "bi-lsd-mnv3l-v1.0.0",
    "opset": OPSET,
    "onnxSizeBytes": os.path.getsize(ONNX_OUT),
    "inputName": INPUT_NAME,
    "outputName": OUTPUT_NAME,
    "inputContract": "raw [0,255] RGB, 1x224x224x3 NHWC float32",
    "outputContract": "[1,1] float32, already a sigmoid probability",
    "tau": TAU,
    "n": int(len(y)),
    "attempts": attempts,
    "versions": {"tensorflow": tf.__version__, "keras": keras.__version__,
                 "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    "parity": {
        "onnx_vs_tflite_max_abs_diff":   float(np.max(np.abs(p_onnx - p_tfl))),
        "onnx_vs_keras_max_abs_diff":    float(np.max(np.abs(p_onnx - p_keras))),
        "keras_vs_tflite_max_abs_diff":  float(np.max(np.abs(p_keras - p_tfl))),
        "onnx_vs_tflite_mean_abs_diff":  float(np.mean(np.abs(p_onnx - p_tfl))),
        "decision_agreement_onnx_tflite": agree(p_onnx, p_tfl),
        "decision_agreement_onnx_keras":  agree(p_onnx, p_keras),
    },
    "metrics_onnx":   metrics(p_onnx),
    "metrics_tflite": metrics(p_tfl),
    "metrics_keras":  metrics(p_keras),
    "expected_from_model_card": {"balanced_accuracy": 0.9090, "auc": 0.9653,
                                 "tp": 389, "tn": 177, "fp": 30, "fn": 15},
    "prob_distribution_onnx": {"min": float(p_onnx.min()),
                               "median": float(np.median(p_onnx)),
                               "max": float(p_onnx.max())},
}

print(json.dumps(report, indent=2))

print("\n" + "=" * 62)
print("GATE: does the ONNX artifact behave as the shipped TFLite does?")
print("=" * 62)
checks = [
    ("decision agreement ONNX vs TFLite == 1.0",
     report["parity"]["decision_agreement_onnx_tflite"] == 1.0),
    ("max abs diff ONNX vs Keras < 1e-4",
     report["parity"]["onnx_vs_keras_max_abs_diff"] < 1e-4),
    ("balanced accuracy within 0.005 of the model card",
     abs(report["metrics_onnx"]["balanced_accuracy"] - 0.9090) < 0.005),
    ("confusion matrix identical to TFLite",
     all(report["metrics_onnx"][k] == report["metrics_tflite"][k]
         for k in ("tp", "tn", "fp", "fn"))),
]
for label, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
report["allChecksPassed"] = all(c[1] for c in checks)
print("\nOVERALL:", "PASS" if report["allChecksPassed"] else "FAIL")

json.dump(report, open("/kaggle/working/onnx_export_report.json", "w"), indent=2)
np.save("/kaggle/working/test_probs_onnx.npy", p_onnx)
np.save("/kaggle/working/test_probs_tflite.npy", p_tfl)

# Keep the output small: the SavedModel dir is scaffolding, not an artifact.
shutil.rmtree(SM, ignore_errors=True)
if os.path.exists(ONNX_RAW):
    os.remove(ONNX_RAW)
print("\noutputs:", sorted(os.listdir("/kaggle/working")))
''')

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (s + "\n").splitlines(keepends=True),
        }
        for s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path("ai/kernels/bovine-lsd-onnx-export/bovine_lsd_onnx_export.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
