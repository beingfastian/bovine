"""Build the browser self-test fixtures.

The Python verification proves the ONNX graph is correct. It does NOT prove the
BROWSER feeds it the same numbers: PIL's bilinear resize and a canvas
drawImage() downscale are different algorithms, and the JPEG decoders differ
too. Those differences land in the pixels, and pixels are exactly where the
predecessor model died.

So: copy a spread of real test images into public/fixtures/ alongside what the
verified model gives them in Python -- lesion probability, species argmax and
P(other). web/app/selftest then runs the same files through the real browser
pipeline and reports the gap as a measured number.

Twelve cattle images span the lesion-probability range (including whatever
sits closest to tau and to both dead-band edges), plus two Pakistani buffalo so
the species head is exercised on both classes it must tell apart.

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
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0-gate-D.onnx"
BUNDLE = ROOT / "ai/data/bovine_lsd_bundle.zip"
BUFFALO = ROOT / "ai/data/_upload_buffalo/buffalo_pak_256.zip"
OUT_DIR = ROOT / "web/public/fixtures"
OUT_JSON = OUT_DIR / "fixtures.json"

N_CATTLE = 12
N_BUFFALO = 2
TAU = 0.50
CLASSES = ["cattle", "buffalo", "other"]


def preprocess(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32)[None, ...]


def main() -> None:
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    assert {"lesion", "species", "ood"} <= set(names), names

    def score(raw: bytes):
        res = dict(zip(names, sess.run(None, {in_name: preprocess(Image.open(io.BytesIO(raw)))})))
        return (float(res["lesion"].ravel()[0]), res["species"].ravel().astype(float),
                float(res["ood"].ravel()[0]))

    # ---------------------------------------------------------------- cattle
    z = zipfile.ZipFile(BUNDLE)
    rows = [r for r in csv.DictReader(io.StringIO(z.read("manifest.csv").decode()))
            if r["split"] == "test"]
    print(f"scoring {len(rows)} cattle test images to choose a spread...")
    scored = []
    for k, r in enumerate(rows, 1):
        raw = z.read(r["file"])
        p, spe, d2 = score(raw)
        scored.append((p, spe, r, raw, d2))
        if k % 100 == 0 or k == len(rows):
            print(f"\r  {k}/{len(rows)}", end="", flush=True)
    print()
    scored.sort(key=lambda t: t[0])

    picks: dict[int, tuple] = {}
    for q in np.linspace(0, len(scored) - 1, N_CATTLE - 3):
        picks[int(round(q))] = scored[int(round(q))]
    for target in (TAU, 0.40, 0.60):
        idx = int(np.argmin([abs(s[0] - target) for s in scored]))
        picks[idx] = scored[idx]

    # --------------------------------------------------------------- buffalo
    bz = zipfile.ZipFile(BUFFALO)
    brows = list(csv.DictReader(io.StringIO(bz.read("manifest.csv").decode())))
    print(f"scoring {len(brows)} buffalo images to pick the clearest {N_BUFFALO}...")
    bscored = []
    for r in brows:
        raw = bz.read(r["file"])
        p, spe, d2 = score(raw)
        bscored.append((spe[1], p, spe, r, raw, d2))
    bscored.sort(key=lambda t: -t[0])
    # one Nili-Ravi (black hide, the model card's weakest breed) and one other
    chosen_b = []
    for pref in ("Neli Ravi", None):
        for cand in bscored:
            if cand in chosen_b:
                continue
            if pref is None or cand[3]["breed"] == pref:
                chosen_b.append(cand)
                break

    # ------------------------------------------------------------------ write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):
        old.unlink()

    fixtures = []
    for p, spe, r, raw, d2 in sorted(picks.values(), key=lambda t: t[0]):
        name = Path(r["file"]).name
        (OUT_DIR / name).write_bytes(raw)
        fixtures.append({
            "file": name, "expectedProbability": round(p, 6),
            "expectedSpecies": CLASSES[int(spe.argmax())], "expectedOther": round(float(spe[2]), 6),
            "expectedOod": round(d2, 3),
            "label": int(r["label"]), "source": r.get("source", ""),
            "severity": r.get("severity", ""), "bytes": len(raw),
        })
        print(f"  {name}  p={p:.4f}  species={CLASSES[int(spe.argmax())]:7} p_other={spe[2]:.3f}  d2={d2:7.1f}  label={r['label']}")
    for _, p, spe, r, raw, d2 in chosen_b:
        name = Path(r["file"]).name
        (OUT_DIR / name).write_bytes(raw)
        fixtures.append({
            "file": name, "expectedProbability": round(p, 6),
            "expectedSpecies": CLASSES[int(spe.argmax())], "expectedOther": round(float(spe[2]), 6),
            "expectedOod": round(d2, 3),
            "label": 0, "source": "buffalo_pak", "severity": "", "breed": r["breed"], "bytes": len(raw),
        })
        print(f"  {name}  p={p:.4f}  species={CLASSES[int(spe.argmax())]:7} p_other={spe[2]:.3f}  {r['breed']}")

    OUT_JSON.write_text(json.dumps({
        "modelVersion": "bi-lsd-mnv3l-v1.0.0-gate-D",
        "generatedBy": "ai/scripts/make_selftest_fixtures.py",
        "note": ("expectedProbability / expectedSpecies / expectedOther are the verified fp32 ONNX model's "
                 "outputs in Python, using PIL bilinear resize to 224x224 on raw [0,255] RGB. The browser "
                 "is expected to differ slightly because canvas resampling is not PIL and the shipped "
                 "weights are fp16; /selftest measures by how much."),
        "tau": TAU, "fixtures": fixtures,
    }, indent=2), encoding="utf-8")
    total = sum(f["bytes"] for f in fixtures)
    print(f"\nwrote {len(fixtures)} fixtures ({total/1024:.0f} kB) -> {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
