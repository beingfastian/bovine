"""A-T1: Inspect the TFLite model's real input/output contract.

Verifies the assumptions AI_IMPLEMENTATION_PLAN.md section 1 makes before any
inference code is written against them.
"""
import sys
from ai_edge_litert.interpreter import Interpreter

MODEL = "bovineinsight/assets/finals_models00.tflite"
LABELS = "bovineinsight/assets/labels.txt"


def main():
    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()

    inp = interp.get_input_details()
    out = interp.get_output_details()

    print("=" * 60)
    print("A-T1  MODEL CONTRACT")
    print("=" * 60)
    for d in inp:
        print(f"INPUT  name={d['name']!r}")
        print(f"       shape={d['shape'].tolist()}  dtype={d['dtype'].__name__}")
        print(f"       quantization={d['quantization']}")
    for d in out:
        print(f"OUTPUT name={d['name']!r}")
        print(f"       shape={d['shape'].tolist()}  dtype={d['dtype'].__name__}")
        print(f"       quantization={d['quantization']}")

    with open(LABELS, encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]
    print(f"\nLABELS ({len(labels)}): {labels}")

    print("\n--- ASSERTIONS ---")
    ishape = inp[0]["shape"].tolist()
    checks = [
        ("input is 4-D NHWC",            len(ishape) == 4),
        ("input is 224x224",             ishape[1] == 224 and ishape[2] == 224),
        ("input has 3 channels (RGB)",   ishape[3] == 3),
        ("input dtype is float32",       inp[0]["dtype"].__name__ == "float32"),
        ("output classes == label count", out[0]["shape"].tolist()[-1] == len(labels)),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed

    # total op count / signature
    try:
        print(f"\nSIGNATURES: {interp.get_signature_list()}")
    except Exception as e:
        print(f"\nSIGNATURES: unavailable ({e})")

    print("\nRESULT:", "A-T1 PASS" if ok else "A-T1 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
