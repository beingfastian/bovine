"""Is the model dead, or is our input convention wrong?

Sweeps candidate preprocessing conventions over a stratified subset and
reports ROC-AUC for each. If EVERY convention gives AUC ~0.5, the weights
carry no signal. If one gives AUC >> 0.5, we simply had the contract wrong.
"""
import io, zipfile
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

ZIP = "ai/data/lumpy_skin_dataset.zip"
N_PER_CLASS = 150

interp = Interpreter(model_path="bovineinsight/assets/finals_models00.tflite")
interp.allocate_tensors()
i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]

CONV = {
    "raw [0,255]":        lambda a: a,
    "/255 -> [0,1]":      lambda a: a / 255.0,
    "/127.5-1 -> [-1,1]": lambda a: a / 127.5 - 1.0,
    "BGR raw [0,255]":    lambda a: a[..., ::-1],
    "BGR /255":           lambda a: (a / 255.0)[..., ::-1],
    "caffe mean-sub":     lambda a: a - np.array([103.939, 116.779, 123.68], np.float32),
}

def auc(y, p):
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p), float); sp = p[o]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j+1] == sp[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2 + 1; i = j + 1
    n1, n0 = int((y==1).sum()), int((y==0).sum())
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0) if n1 and n0 else float("nan")

zf = zipfile.ZipFile(ZIP)
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
lump = [n for n in names if "lumpy" in n.lower()][:N_PER_CLASS]
norm = [n for n in names if "normal" in n.lower()][:N_PER_CLASS]
sel = [(n, 1) for n in lump] + [(n, 0) for n in norm]
print(f"subset: {len(lump)} lumpy + {len(norm)} normal\n")

arrs, ys = [], []
for n, lab in sel:
    im = Image.open(io.BytesIO(zf.read(n))).convert("RGB").resize((224,224), Image.BILINEAR)
    arrs.append(np.asarray(im, np.float32)); ys.append(lab)
y = np.array(ys)

print(f"{'convention':22} {'AUC':>7} {'min':>9} {'max':>9} {'mean':>9} {'#distinct':>10}")
print("-" * 72)
for name, fn in CONV.items():
    ps = []
    for a in arrs:
        x = np.ascontiguousarray(fn(a.copy())[None, ...], dtype=np.float32)
        interp.set_tensor(i_d["index"], x); interp.invoke()
        ps.append(float(interp.get_tensor(o_d["index"])[0][0]))
    p = np.array(ps)
    print(f"{name:22} {auc(y,p):7.4f} {p.min():9.5f} {p.max():9.5f} {p.mean():9.5f} {len(np.unique(p)):10d}")

print("\n>> If '#distinct' is 1 for every convention, the output is CONSTANT: dead model.")
