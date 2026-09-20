"""Build the browser self-test fixtures.

The Python verification proves the ONNX graph is correct. It does NOT prove the
BROWSER feeds it the same numbers: PIL's bilinear resize and a canvas
drawImage() downscale are different algorithms, and the JPEG decoders differ
too. Those differences land in the pixels, and pixels are exactly where the
predecessor model died.

So: copy a spread of real test images into public/fixtures/ alongside the
probability the verified ONNX model gives them in Python. web/app/selftest then
runs the same files through the real browser pipeline and reports the gap as a
measured number.

Run:  python ai/scripts/make_selftest_fixtures.py
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.onnx"
BUNDLE = ROOT / "ai/data/bovine_lsd_bundle.zip"
OUT_DIR = ROOT / "web/public/fixtures"
OUT_JSON = OUT_DIR / "fixtures.json"

N_FIXTURES = 12
TAU = 0.50


def preprocess(im: Image.Image) -> np.ndarray:
    return np.asarray(
        im.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32
    )[None, ...]


def main() -> None:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    z = zipfile.ZipFile(BUNDLE)
    rows = [
        r
        for r in csv.DictReader(io.StringIO(z.read("manifest.csv").decode()))
        if r["split"] == "test"
    ]

    print(f"scoring {len(rows)} test images to choose a spread...")
    scored = []
    for k, r in enumerate(rows, 1):
        raw = z.read(r["file"])
        p = float(sess.run(None, {in_name: preprocess(Image.open(io.BytesIO(raw)))})[0].ravel()[0])
        scored.append((p, r, raw))
        if k % 100 == 0 or k == len(rows):
            print(f"\r  {k}/{len(rows)}", end="", flush=True)
    print()

    scored.sort(key=lambda t: t[0])

    # Spread across the probability range, and deliberately include whatever
    # sits closest to tau and to the dead-band edges -- those are the samples
    # where a small preprocessing drift would actually flip a verdict.
    picks: dict[int, tuple] = {}
    for q in np.linspace(0, len(scored) - 1, N_FIXTURES - 3):
        picks[int(round(q))] = scored[int(round(q))]
    for target in (TAU, 0.40, 0.60):
        idx = int(np.argmin([abs(s[0] - target) for s in scored]))
        picks[idx] = scored[idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):
        old.unlink()

    fixtures = []
    for p, r, raw in sorted(picks.values(), key=lambda t: t[0]):
        name = Path(r["file"]).name
        (OUT_DIR / name).write_bytes(raw)
        fixtures.append(
            {
                "file": name,
                "expectedProbability": round(p, 6),
                "label": int(r["label"]),
                "source": r.get("source", ""),
                "severity": r.get("severity", ""),
                "bytes": len(raw),
            }
        )
        print(f"  {name}  p={p:.4f}  label={r['label']}  {r.get('source','')}")

    OUT_JSON.write_text(
        json.dumps(
            {
                "modelVersion": "bi-lsd-mnv3l-v1.0.0",
                "generatedBy": "ai/scripts/make_selftest_fixtures.py",
                "note": (
                    "expectedProbability is the verified ONNX model's output in "
                    "Python, using PIL bilinear resize to 224x224 on raw [0,255] "
                    "RGB. The browser is expected to differ slightly because "
                    "canvas resampling is not PIL; /selftest measures by how much."
                ),
                "tau": TAU,
                "fixtures": fixtures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(f["bytes"] for f in fixtures)
    print(f"\nwrote {len(fixtures)} fixtures ({total/1024:.0f} kB) -> {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
