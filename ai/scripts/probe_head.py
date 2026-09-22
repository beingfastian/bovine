"""Determine whether the single output is a raw logit or an activated probability."""
import numpy as np, re
from ai_edge_litert.interpreter import Interpreter

MODEL = "bovineinsight/assets/finals_models00.tflite"

# 1. Look for a sigmoid (LOGISTIC) op in the graph
blob = open(MODEL, "rb").read()
names = {n.decode(errors="ignore") for n in re.findall(rb"[ -~]{4,}", blob)}
tail = sorted(n for n in names if any(k in n.lower() for k in
              ("sigmoid", "logistic", "softmax", "dense_1_2", "dense_2")))
print("TAIL-OF-GRAPH OP NAMES:")
for n in tail:
    print("  ", n[:150])

# 2. Empirical: feed varied inputs, see the output range
interp = Interpreter(model_path=MODEL)
interp.allocate_tensors()
i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]

rng = np.random.default_rng(0)
cases = {
    "all zeros (black)":      np.zeros((1,224,224,3), np.float32),
    "all 255 (white)":        np.full((1,224,224,3), 255.0, np.float32),
    "uniform random 0-255":   rng.uniform(0,255,(1,224,224,3)).astype(np.float32),
    "uniform random 0-1":     rng.uniform(0,1,(1,224,224,3)).astype(np.float32),
    "gaussian mean128":       np.clip(rng.normal(128,60,(1,224,224,3)),0,255).astype(np.float32),
    "mid grey 128":           np.full((1,224,224,3), 128.0, np.float32),
}
print("\nOUTPUT FOR SYNTHETIC INPUTS:")
vals = []
for name, x in cases.items():
    interp.set_tensor(i_d["index"], x)
    interp.invoke()
    v = float(interp.get_tensor(o_d["index"])[0][0])
    vals.append(v)
    print(f"  {name:24} -> {v:+.6f}")

lo, hi = min(vals), max(vals)
print(f"\nrange observed: [{lo:.6f}, {hi:.6f}]")
if 0.0 <= lo and hi <= 1.0:
    print(">> All outputs in [0,1]: sigmoid is very likely BAKED IN (value = P(class 1))")
else:
    print(">> Outputs escape [0,1]: this is a RAW LOGIT -> apply sigmoid yourself")
