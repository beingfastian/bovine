"""Step 2 — Deduplicate and build the training manifest.

Runs on CPU. No GPU, no TensorFlow. Reads zips directly (no extraction).

WHY THIS EXISTS
---------------
Improper splits inflate reported accuracy by 5-30% on medical image datasets
(Nature Scientific Data, OCT study). The LSD literature shows exactly this
signature: ConvNeXtLSD reports 98.29% on its test split but 88.31% under
cross-validation. Near-duplicate images straddling train/test are the cause.

THREE LAYERS OF DEDUPLICATION
-----------------------------
1. Roboflow augmentation stripping -- "<stem>_jpg.rf.<hash>.jpg" files are
   augmented copies of ONE source image. Deterministic: keep one per stem.
2. Exact duplicates (SHA-256 over raw bytes).
3. Near-duplicates (perceptual hash + union-find clustering).

Layer 1 matters because pHash does NOT reliably collapse flips and rotations,
so augmented siblings would survive as separate "groups" and leak across folds.
The Roboflow v11 export is 8,014 files from only ~2,784 source images.

MEMORY
------
Full image bytes are hashed then discarded; only a 256px thumbnail is retained.
Keeps ~13k images well inside RAM.

OUTPUT
------
ai/data/manifest.csv   path, label, source, sha256, phash, group_id
ai/reports/dedupe.md   the numbers you report
ai/reports/dupe_pairs/ sample matched pairs -- LOOK AT THESE to tune THRESH

USAGE
-----
    python ai/scripts/dedupe.py
    python ai/scripts/dedupe.py --thresh 3      # stricter
    python ai/scripts/dedupe.py --thresh 8      # looser
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    raise SystemExit("pip install imagehash")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai" / "data"
REPORTS = ROOT / "ai" / "reports"

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
THUMB = 256

# Roboflow augmented-copy filename pattern: <stem>_jpg.rf.<md5>.jpg
RF_AUG = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", re.I)


# ---------------------------------------------------------------- sources
# label: 1 = lesion/lumpy, 0 = normal/healthy, None = skip (OOD holdout)
SOURCES = [
    {
        "name": "mendeley",
        "path": DATA / "lumpy_skin_dataset.zip",
        "format": "folders",
        "rules": [("lumpy", 1), ("normal", 0)],
    },
    {
        "name": "roboflow",
        "path": DATA / "cattle_disease_yolov8.zip",
        "format": "yolo",
        "rules": [
            ("lumpy", 1),
            ("normal_healthy_cow", 0),
            # HELD OUT for the T6 OOD suite -- deliberately not trained on:
            ("infected_foot", None),
            ("mouth_disease", None),
            ("normal_mouth", None),
        ],
    },
    # Zenodo LumpySkinDisease_DataHub is NOT listed here: verified byte-identical
    # to the Mendeley set (250/250 sha256 matches). It contributes 0 new images.
    # Its value is SEVERITY labels (Normal/Mild/Severe) -> ai/data/severity_labels.csv,
    # joined onto the manifest by sha256 for stratified evaluation.
]


def load_overrides() -> dict:
    """Manual label decisions, applied after loading. Decisions as data, not edits."""
    f = DATA / "label_overrides.csv"
    if not f.exists():
        return {}
    out = {}
    with open(f, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["match"].strip()] = (
                row["action"].strip(),
                int(row["label"]) if row["label"].strip() else None,
            )
    return out


def classify(name: str, rules) -> int | None:
    """Map a name to a label using the first matching rule. None = skip."""
    low = name.lower().replace(" ", "_").replace("-", "_")
    for key, label in rules:
        if key in low:
            return label
    return None


def source_stem(fname: str) -> str:
    """Collapse Roboflow augmented copies onto their source image stem."""
    return RF_AUG.sub("", fname) or fname


def digest(blob: bytes):
    """Return (sha256, thumbnail). Caller drops the full blob afterwards."""
    sha = hashlib.sha256(blob).hexdigest()
    im = Image.open(io.BytesIO(blob))
    try:
        im.draft("RGB", (THUMB, THUMB))      # fast JPEG downscale during decode
    except Exception:
        pass
    im = im.convert("RGB")
    im.thumbnail((THUMB, THUMB), Image.BILINEAR)
    return sha, im


# ---------------------------------------------------------------- readers
def read_folders(src) -> list[dict]:
    """Class-per-folder layout, zip or directory."""
    p: Path = src["path"]
    out = []
    if not p.exists():
        print(f"  !! MISSING: {p.name}  (skipping)")
        return out

    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            for n in zf.namelist():
                if n.endswith("/") or "__MACOSX" in n:
                    continue
                if not n.lower().endswith(IMG_EXT):
                    continue
                lab = classify(n, src["rules"])
                if lab is None:
                    continue
                out.append({"path": f"{p.name}::{n}", "label": lab,
                            "blob": zf.read(n), "stem": Path(n).name})
    else:
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                lab = classify(str(f.relative_to(p)), src["rules"])
                if lab is not None:
                    out.append({"path": str(f.relative_to(ROOT)), "label": lab,
                                "blob": f.read_bytes(), "stem": f.name})
    return out


def read_yolo(src) -> list[dict]:
    """Roboflow OBJECT-DETECTION export (YOLO layout), zip or directory.

    Image-level label from the box classes present:
      any lesion box  -> 1
      else normal box -> 0
      else            -> None (held out for the T6 OOD suite)
    """
    p: Path = src["path"]
    out = []
    if not p.exists():
        print(f"  !! MISSING: {p.name}  (skipping)")
        return out

    def parse_names(text: str):
        m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",")]
        return re.findall(r"^\s*-\s*(.+?)\s*$", text.split("names:")[1], re.M)

    def label_for(idxs, names):
        lab = None
        for i in idxs:
            if i >= len(names):
                continue
            m = classify(names[i], src["rules"])
            if m == 1:
                return 1
            if m == 0 and lab is None:
                lab = 0
        return lab

    n_ood = 0
    with zipfile.ZipFile(p) as zf:
        yml = next((n for n in zf.namelist() if n.endswith("data.yaml")), None)
        if yml is None:
            print(f"  !! no data.yaml in {p.name}")
            return out
        names = parse_names(zf.read(yml).decode("utf-8"))
        print(f"  data.yaml classes: {names}")
        labelset = {n for n in zf.namelist() if "/labels/" in n and n.endswith(".txt")}
        for n in zf.namelist():
            if "/images/" not in n or not n.lower().endswith(IMG_EXT):
                continue
            lt = n.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            idxs = set()
            if lt in labelset:
                for line in zf.read(lt).decode().splitlines():
                    if line.strip():
                        idxs.add(int(line.split()[0]))
            lab = label_for(idxs, names)
            if lab is None:
                n_ood += 1
                continue
            out.append({"path": f"{p.name}::{n}", "label": lab,
                        "blob": zf.read(n), "stem": source_stem(Path(n).name)})

    print(f"  held out for OOD suite (foot/mouth/unlabelled): {n_ood}")
    return out


# ---------------------------------------------------------------- clustering
def hamming_clusters(bits: np.ndarray, thresh: int) -> list[list[int]]:
    n = len(bits)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    CHUNK = 256
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        d = (bits[i0:i1, None, :] != bits[None, :, :]).sum(axis=2)
        for i in range(i1 - i0):
            for j in np.nonzero(d[i] <= thresh)[0]:
                if j > i0 + i:
                    union(i0 + i, int(j))
        print(f"\r  clustering {i1}/{n}", end="", flush=True)
    print()

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=20)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ 1. load, hash, discard
    print("STEP 2.1  loading sources")
    records, stats = [], {}
    for src in SOURCES:
        items = read_yolo(src) if src["format"] == "yolo" else read_folders(src)
        raw = len(items)
        if raw == 0:
            stats[src["name"]] = (0, 0)
            continue

        # --- LAYER 1: strip augmented copies, deterministically by stem ---
        seen, kept = set(), []
        for r in items:
            key = (src["name"], r["stem"])
            if key in seen:
                continue
            seen.add(key)
            kept.append(r)
        if raw != len(kept):
            print(f"  augmented copies stripped: {raw - len(kept)}")

        for r in kept:
            r["sha256"], r["thumb"] = digest(r.pop("blob"))
            r["source"] = src["name"]
        records.extend(kept)
        stats[src["name"]] = (raw, len(kept))
        print(f"  {src['name']:12} {raw:6d} files -> {len(kept):6d} source images\n")

    if not records:
        raise SystemExit("no images found -- check SOURCES paths")

    # ------------------------------------------------ overrides
    ov = load_overrides()
    if ov:
        dropped = relabelled = 0
        kept = []
        for r in records:
            base = r["path"].split("::")[-1].split("/")[-1]
            act = ov.get(base)
            if act is None:
                kept.append(r)
                continue
            action, lab = act
            if action == "drop":
                dropped += 1
                continue
            if action == "relabel":
                relabelled += r["label"] != lab
                r["label"] = lab
            kept.append(r)
        records = kept
        print(f"  overrides: {dropped} dropped, {relabelled} relabelled")

    n_raw = len(records)
    print(f"  TOTAL: {n_raw}  "
          f"({sum(r['label'] for r in records)} lesion / "
          f"{sum(1 - r['label'] for r in records)} normal)\n")

    # ------------------------------------------------ 2. exact duplicates
    print("STEP 2.2  exact duplicates (SHA-256)")
    by_sha = defaultdict(list)
    for r in records:
        by_sha[r["sha256"]].append(r)
    n_exact = sum(len(v) - 1 for v in by_sha.values() if len(v) > 1)
    print(f"  exact duplicate files: {n_exact}")

    conflicts = [v for v in by_sha.values() if len({r['label'] for r in v}) > 1]
    print(f"  !! CONFLICTING LABELS: {len(conflicts)}")
    for v in conflicts[:8]:
        print("      " + " | ".join(f"{r['path'][-46:]}->{r['label']}" for r in v))

    cross = sum(1 for v in by_sha.values() if len({r["source"] for r in v}) > 1)
    print(f"  CROSS-SOURCE exact duplicates: {cross}")

    uniq = [v[0] for v in by_sha.values()]
    print(f"  after exact dedupe: {len(uniq)}\n")

    # ------------------------------------------------ 3. near duplicates
    print(f"STEP 2.3  near-duplicates (pHash, Hamming <= {args.thresh})")
    bits, keep = [], []
    for k, r in enumerate(uniq, 1):
        try:
            h = imagehash.phash(r["thumb"])
        except Exception as e:
            print(f"\n  unhashable: {r['path']} ({e})")
            continue
        r["phash"] = str(h)
        bits.append(h.hash.flatten())
        keep.append(r)
        if k % 500 == 0:
            print(f"\r  hashing {k}/{len(uniq)}", end="", flush=True)
    print()

    clusters = hamming_clusters(np.array(bits, dtype=bool), args.thresh)
    multi = [c for c in clusters if len(c) > 1]
    in_cluster = sum(len(c) for c in multi)
    print(f"  near-duplicate clusters: {len(multi)}")
    print(f"  images inside a cluster: {in_cluster}")

    for gid, members in enumerate(clusters):
        for i in members:
            keep[i]["group_id"] = gid
    n_groups = len(clusters)

    xsrc = sum(1 for c in clusters if len({keep[i]["source"] for i in c}) > 1)
    print(f"  CROSS-SOURCE near-dup clusters: {xsrc}")
    print(f"\n  >>> TRUE DATASET SIZE: {n_groups} groups (not {n_raw} files) <<<\n")

    # ------------------------------------------------ 4. sample pairs
    outdir = REPORTS / "dupe_pairs"
    outdir.mkdir(exist_ok=True)
    for f in outdir.glob("*.png"):
        f.unlink()
    shown = 0
    for c in multi:
        if shown >= args.pairs:
            break
        a, b = keep[c[0]], keep[c[1]]
        sheet = Image.new("RGB", (452, 224), "white")
        sheet.paste(a["thumb"].resize((224, 224)), (0, 0))
        sheet.paste(b["thumb"].resize((224, 224)), (228, 0))
        tag = "SAMELABEL" if a["label"] == b["label"] else "DIFFLABEL"
        xs = "XSRC" if a["source"] != b["source"] else "same"
        sheet.save(outdir / f"pair_{shown:02d}_{tag}_{xs}.png")
        shown += 1
    print(f"STEP 2.4  wrote {shown} sample pairs -> {outdir}\n")

    # ------------------------------------------------ 5. manifest
    man = DATA / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "sha256", "phash", "group_id"])
        for r in keep:
            w.writerow([r["path"], r["label"], r["source"],
                        r["sha256"], r["phash"], r["group_id"]])
    print(f"wrote {man}  ({len(keep)} rows)")

    gl = defaultdict(set)
    for r in keep:
        gl[r["group_id"]].add(r["label"])
    impure = [g for g, ls in gl.items() if len(ls) > 1]
    print(f"groups with MIXED labels: {len(impure)}")

    lesion_g = len({r["group_id"] for r in keep if r["label"] == 1})
    normal_g = len({r["group_id"] for r in keep if r["label"] == 0})
    print(f"\nlesion groups: {lesion_g}   normal groups: {normal_g}   "
          f"imbalance {normal_g / max(1, lesion_g):.2f}:1")

    # ------------------------------------------------ 6. report
    srctab = "\n".join(f"| {k} | {v[0]} | {v[1]} |" for k, v in stats.items())
    rep = f"""# Step 2 — Deduplication Report

**pHash threshold:** {args.thresh} (Hamming)

## Per source

| Source | Raw files | After augmentation-stripping |
|---|---|---|
{srctab}

## Deduplication

| Metric | Value |
|---|---|
| Source images in | {n_raw} |
| Exact duplicates (SHA-256) | {n_exact} |
| **Cross-source exact duplicates** | **{cross}** |
| Conflicting labels | {len(conflicts)} |
| Near-duplicate clusters | {len(multi)} |
| Images inside a cluster | {in_cluster} |
| **Cross-source near-dup clusters** | **{xsrc}** |
| **TRUE dataset size (groups)** | **{n_groups}** |
| Lesion groups | {lesion_g} |
| Normal groups | {normal_g} |
| Imbalance | {normal_g / max(1, lesion_g):.2f}:1 |
| Groups with mixed labels | {len(impure)} |

Report `{n_groups}` as the dataset size. Never the file count.

`group_id` in `ai/data/manifest.csv` goes to `StratifiedGroupKFold(groups=...)`.
An entire group belongs to exactly one fold.
"""
    (REPORTS / "dedupe.md").write_text(rep, encoding="utf-8")
    print(f"wrote {REPORTS / 'dedupe.md'}")


if __name__ == "__main__":
    main()
