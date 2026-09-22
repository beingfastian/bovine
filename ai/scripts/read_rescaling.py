"""Read the model's built-in Rescaling constants straight from the flatbuffer.

Decides the input convention definitively instead of inferring it.
"""
import numpy as np
from ai_edge_litert import schema_py_generated as schema

MODEL = "bovineinsight/assets/finals_models00.tflite"

buf = bytearray(open(MODEL, "rb").read())
model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
sg = model.subgraphs[0]

opcodes = []
for oc in model.operatorCodes:
    code = oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode
    name = next((k for k, v in vars(schema.BuiltinOperator).items()
                 if not k.startswith("_") and v == code), f"UNKNOWN({code})")
    opcodes.append(name)

def tname(i):
    n = sg.tensors[i].name
    return n.decode() if isinstance(n, (bytes, bytearray)) else str(n)

def const(i):
    t = sg.tensors[i]
    b = model.buffers[t.buffer]
    if b.data is None or len(b.data) == 0:
        return None
    dt = {0: np.float32, 2: np.int32, 3: np.uint8, 9: np.int8}.get(t.type, np.float32)
    return np.frombuffer(bytes(b.data), dtype=dt)

print("First 8 ops in the graph:\n")
for k, op in enumerate(sg.operators[:8]):
    name = opcodes[op.opcodeIndex]
    print(f"[{k}] {name}")
    for i in op.inputs:
        c = const(i)
        cs = "input/activation" if c is None else f"CONST {c[:4]}{'...' if c.size>4 else ''} (n={c.size})"
        print(f"      in  {tname(i)[:70]:70} {cs}")
    for o in op.outputs:
        print(f"      out {tname(o)[:70]}")
    print()

print("\n" + "="*60)
print("LAST 6 OPS (determines output activation)")
print("="*60)
n = len(sg.operators)
for k, op in enumerate(sg.operators[-6:], start=n-6):
    name = opcodes[op.opcodeIndex]
    print(f"[{k}/{n-1}] {name}")
    for o in op.outputs:
        print(f"       out {tname(o)[:80]}")
print("\nGraph outputs:", [tname(i) for i in sg.outputs])
print("\nAll distinct opcodes used:", sorted(set(opcodes)))
