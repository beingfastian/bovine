"""v1.1 — build a LESION-CROP training bundle from the Roboflow bounding boxes.

WHY
---
v1.0.0 Grad-CAM showed broad, whole-body activation rather than focal attention
on nodules. That predicted and explained the severity gap (Severe recall 0.99 vs
Mild 0.75): the model reads "this animal looks generally affected" instead of
"these are discrete nodules".

Training on lesion CROPS forces focal learning. The Roboflow export ships YOLO
boxes we never used: 4,243 lesion boxes across 1,515 source images.

DESIGN DECISIONS
----------------
1. CONTEXT PADDING (25%). A lesion crop with zero margin loses the surrounding
   normal skin that makes a nodule read as a nodule.

2. SQUARE CROPS. Expanding the shorter side avoids aspect distortion on resize.

3. NEGATIVES MATCH THE POSITIVE SIZE DISTRIBUTION. This is the critical one.
   If positives were small crops and negatives large ones, the model would learn
   CROP SCALE instead of pathology -- the same class of shortcut as the
   source-recognition problem in v1.0.0. Each negative samples its box size from
   the empirical positive distribution.

4. GROUPS ARE INHERITED FROM THE PARENT IMAGE. Every crop of one photo carries
   that photo's group_id and split, so crops of the same animal can never
   straddle train/test.

5. MIN AREA FLOOR 0.5%. Below that a box is degenerate. Boxes between 0.5% and
   2% are KEPT -- they are the small early lesions this version exists to catch.

    python ai/scripts/build_crops.py
    -> ai/data/bovine_lsd_crops.zip
"""
from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai" / "data"

RF_ZIP = DATA / "cattle_disease_yolov8.zip"
MEN_ZIP = DATA / "lumpy_skin_dataset.zip"
OUT = DATA / "bovine_lsd_crops.zip"

SEED = 42
PAD = 0.25            # context margin around each box
MIN_AREA = 0.005      # drop degenerate boxes
SIZE = 256
QUALITY = 92
NEG_PER_IMAGE = 4     # random crops sampled per healthy image

RF_AUG = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+\.[A-Za-z]+$", re.I)


def stem_of(fname: str) -> str:
    return RF_AUG.sub("", fname)


def square_crop(im: Image.Image, cx, cy, w, h, pad=PAD):
    """Padded square crop around a normalised YOLO box, clamped to the image."""
    W, H = im.size
    side = max(w * W, h * H) * (1 + 2 * pad)
    side = max(side, 24)
    x0 = cx * W - side / 2
    y0 = cy * H - side / 2
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = min(W, x0 + side)
    y1 = min(H, y0 + side)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


def encode(im: Image.Image) -> bytes:
    im = im.convert("RGB")
    im.thumbnail((SIZE, SIZE), Image.LANCZOS)
    if min(im.size) < 64:
        return b""
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=QUALITY, optimize=True)
    return buf.getvalue()


def main():
    rng = random.Random(SEED)

    # ---------------------------------------------- group/split lookup
    rows = list(csv.DictReader(open(DATA / "train_manifest.csv", encoding="utf-8")))
    key2meta = {}
    for r in rows:
        fname = r["path"].split("::")[-1].split("/")[-1]
        key2meta[(r["source"], stem_of(fname))] = (r["group_id"], r["split"])
    print(f"manifest lookup: {len(key2meta)} source images")

    rf = zipfile.ZipFile(RF_ZIP)
    yml = next(n for n in rf.namelist() if n.endswith("data.yaml"))
    names = [x.strip().strip("'\"") for x in
             re.search(r"names:\s*\[(.*?)\]", rf.read(yml).decode(), re.S)
             .group(1).split(",")]
    LUMPY = {i for i, n in enumerate(names) if "lumpy" in n.lower()}
    HEALTHY = {i for i, n in enumerate(names)
               if "normal_healthy" in n.lower().replace(" ", "_")}
    print(f"classes: {names}\n  lumpy={LUMPY}  healthy={HEALTHY}")

    img_by_stem = {}
    for n in rf.namelist():
        if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png")):
            img_by_stem.setdefault(stem_of(n.split("/")[-1]), n)

    # ---------------------------------------------- pass 1: positives
    out_rows, pos_sizes, skipped = [], [], Counter()
    seen, emitted = set(), set()
    written = 0

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:

        for n in rf.namelist():
            if "/labels/" not in n or not n.endswith(".txt"):
                continue
            stem = stem_of(n.split("/")[-1])
            if stem in seen:
                continue
            seen.add(stem)

            lines = [l.split() for l in rf.read(n).decode().splitlines() if l.strip()]
            boxes = [(float(a), float(b), float(c), float(d))
                     for i, a, b, c, d in
                     ((int(l[0]), l[1], l[2], l[3], l[4]) for l in lines)
                     if i in LUMPY]
            if not boxes:
                continue

            meta = key2meta.get(("roboflow", stem))
            if meta is None:
                skipped["no manifest entry"] += 1
                continue
            gid, split = meta

            img_name = img_by_stem.get(stem)
            if img_name is None:
                skipped["no image"] += 1
                continue
            im = Image.open(io.BytesIO(rf.read(img_name)))

            for k, (cx, cy, w, h) in enumerate(boxes):
                if w * h < MIN_AREA:
                    skipped["below area floor"] += 1
                    continue
                c = square_crop(im, cx, cy, w, h)
                if c is None:
                    skipped["crop too small"] += 1
                    continue
                data = encode(c)
                if not data:
                    skipped["encode too small"] += 1
                    continue
                # content hash, NOT a truncated stem: IMG-20220830-WA0001 and
                # ...WA0002 truncate identically and silently collapse crops
                # from DIFFERENT groups onto one filename.
                fn = f"images/pos_{hashlib.sha256(data).hexdigest()[:20]}.jpg"
                if fn in emitted:
                    skipped["identical crop"] += 1
                    continue
                emitted.add(fn)
                zo.writestr(fn, data)
                out_rows.append(dict(file=fn, label=1, source="roboflow",
                                     group_id=gid, split=split, kind="lesion_box",
                                     box_area=round(w * h, 6),
                                     box_side=round(max(w, h), 6)))
                pos_sizes.append(max(w, h))
                written += 1
            if written and written % 500 == 0:
                print(f"\r  positives {written}", end="", flush=True)
        print(f"\n  positives written: {written}")
        if not pos_sizes:
            raise SystemExit("no positive crops produced -- check paths")
        pos_sizes = np.array(pos_sizes)

        # ------------------------------------------ pass 2: negatives
        # Sample crop scale from the POSITIVE distribution so the model cannot
        # learn crop size as a proxy for the label.
        def neg_crops(im, gid, split, source, tag):
            made = 0
            for j in range(NEG_PER_IMAGE):
                s = float(rng.choice(pos_sizes))
                cx = rng.uniform(s / 2, 1 - s / 2) if s < 1 else 0.5
                cy = rng.uniform(s / 2, 1 - s / 2) if s < 1 else 0.5
                c = square_crop(im, cx, cy, s, s)
                if c is None:
                    continue
                data = encode(c)
                if not data:
                    continue
                fn = f"images/neg_{hashlib.sha256(data).hexdigest()[:20]}.jpg"
                if fn in emitted:
                    continue
                emitted.add(fn)
                zo.writestr(fn, data)
                out_rows.append(dict(file=fn, label=0, source=source,
                                     group_id=gid, split=split,
                                     kind="healthy_patch",
                                     box_area=round(s * s, 6),
                                     box_side=round(s, 6)))
                made += 1
            return made

        n_neg = 0
        # (a) Roboflow healthy-cow images
        seen2 = set()
        for n in rf.namelist():
            if "/labels/" not in n or not n.endswith(".txt"):
                continue
            stem = stem_of(n.split("/")[-1])
            if stem in seen2:
                continue
            seen2.add(stem)
            lines = [l.split() for l in rf.read(n).decode().splitlines() if l.strip()]
            idxs = {int(l[0]) for l in lines}
            if not (idxs & HEALTHY) or (idxs & LUMPY):
                continue
            meta = key2meta.get(("roboflow", stem))
            if meta is None:
                continue
            img_name = img_by_stem.get(stem)
            if img_name is None:
                continue
            im = Image.open(io.BytesIO(rf.read(img_name)))
            n_neg += neg_crops(im, meta[0], meta[1], "roboflow", f"rf_{stem[:14]}")
        print(f"  negatives from roboflow healthy: {n_neg}")

        # (b) Mendeley "Normal Skin" images
        men = zipfile.ZipFile(MEN_ZIP)
        n_men = 0
        for n in men.namelist():
            if not n.lower().endswith(".png") or "normal" not in n.lower():
                continue
            fname = n.split("/")[-1]
            meta = key2meta.get(("mendeley", fname))
            if meta is None:
                continue
            im = Image.open(io.BytesIO(men.read(n)))
            n_men += neg_crops(im, meta[0], meta[1], "mendeley",
                               f"men_{fname.rsplit('.',1)[0][:16]}")
        print(f"  negatives from mendeley normal : {n_men}")

        # ------------------------------------------ manifest
        sio = io.StringIO()
        w_ = csv.DictWriter(sio, fieldnames=list(out_rows[0].keys()))
        w_.writeheader(); w_.writerows(out_rows)
        zo.writestr("manifest.csv", sio.getvalue())
        zo.writestr("README.txt",
                    "BovineInsight v1.1 LESION-CROP bundle\n"
                    f"positives: lesion bounding-box crops, {PAD:.0%} context padding\n"
                    "negatives: random patches from healthy animals, crop scale\n"
                    "           sampled from the POSITIVE size distribution so that\n"
                    "           crop size cannot act as a label proxy\n"
                    "group_id/split inherited from the parent photo -- crops of one\n"
                    "animal never straddle train/test\n"
                    "Sources: Roboflow cattle_disease-detection (CC BY 4.0),\n"
                    "         Mendeley w36hpf86j2 (CC BY 4.0)\n")

    # ---------------------------------------------- report
    lab = Counter(r["label"] for r in out_rows)
    spl = Counter((r["split"], r["label"]) for r in out_rows)
    size = OUT.stat().st_size
    print(f"\nwrote {OUT}  ({size/1e6:.1f} MB)")
    print(f"  crops total : {len(out_rows)}")
    print(f"  lesion      : {lab[1]}")
    print(f"  healthy     : {lab[0]}")
    print(f"  ratio       : {lab[0]/max(1,lab[1]):.2f} : 1")
    if skipped:
        print("  skipped:", dict(skipped))
    print("\n  split composition:")
    for k in sorted(spl):
        print(f"    {k[0]:8} label={k[1]}  {spl[k]:5d}")
    groups = {(r["group_id"], r["split"]) for r in out_rows}
    gsplit = defaultdict(set)
    for g, s in groups:
        gsplit[g].add(s)
    bad = [g for g, s in gsplit.items() if len(s) > 1]
    print(f"\n  groups spanning >1 split: {len(bad)}  "
          f"{'[PASS]' if not bad else '[FAIL]'}")


    import numpy as _np
    pos = [r for r in out_rows if r["label"] == 1]
    ar = _np.array([r["box_area"] for r in pos])
    q1 = float(_np.percentile(ar, 25))
    print()
    print(f"  lesion box area: p25={q1:.4f}  med={_np.median(ar):.4f}  "
          f"p75={_np.percentile(ar,75):.4f}")
    small = Counter(r["split"] for r in pos if r["box_area"] <= q1)
    print("  SMALL lesions (bottom quartile by area) per split -- the v1.1 target:")
    for k in sorted(small):
        print(f"    {k:8} {small[k]:5d}")

    files = Counter(r["file"] for r in out_rows)
    dups = {k: v for k, v in files.items() if v > 1}
    print(f"  duplicate filenames: {len(dups)}  ", "[PASS]" if not dups else "[FAIL]")
    assert not dups, f"{len(dups)} duplicate filenames -- manifest corrupt"
    assert not bad, "group leak across splits"


if __name__ == "__main__":
    main()
