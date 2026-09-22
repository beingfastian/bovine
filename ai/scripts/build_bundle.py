"""Build a compact Kaggle upload bundle.

The raw sources are ~1 GB (Mendeley 155 MB + Roboflow 860 MB). Uploading that
over a slow link is painful and unnecessary: training runs at 224x224.

This resizes every manifested image to max 256 px (leaving headroom for
RandomZoom/crop down to 224) and packs them with the manifest into one zip,
typically 40-80 MB.

Images are re-keyed by SHA-256 prefix so filenames are stable, collision-free,
and carry no label information that could leak through a filename.

    python ai/scripts/build_bundle.py
    -> ai/data/bovine_lsd_bundle.zip
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai" / "data"

SIZE = 256
QUALITY = 92
OUT = DATA / "bovine_lsd_bundle.zip"


def main():
    rows = list(csv.DictReader(open(DATA / "train_manifest.csv", encoding="utf-8")))
    print(f"manifest: {len(rows)} images")

    # open each source zip once
    zips: dict[str, zipfile.ZipFile] = {}

    def read_image(path: str) -> bytes:
        if "::" in path:
            zname, member = path.split("::", 1)
            if zname not in zips:
                zips[zname] = zipfile.ZipFile(DATA / zname)
            return zips[zname].read(member)
        return (ROOT / path).read_bytes()

    n_bytes_in = n_bytes_out = 0
    failed = []
    stats = Counter()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        w_rows = []
        for i, r in enumerate(rows, 1):
            try:
                blob = read_image(r["path"])
            except Exception as e:
                failed.append((r["path"], str(e)))
                continue
            n_bytes_in += len(blob)

            im = Image.open(io.BytesIO(blob))
            try:
                im.draft("RGB", (SIZE, SIZE))
            except Exception:
                pass
            im = im.convert("RGB")
            im.thumbnail((SIZE, SIZE), Image.LANCZOS)

            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=QUALITY, optimize=True)
            data = buf.getvalue()
            n_bytes_out += len(data)

            fname = f"images/{r['sha256'][:16]}.jpg"
            out.writestr(fname, data)
            stats[(r["split"], r["label"])] += 1

            w_rows.append({
                "file": fname,
                "label": r["label"],
                "source": r["source"],
                "group_id": r["group_id"],
                "split": r["split"],
                "severity": r["severity"],
                "sha256": r["sha256"],
            })
            if i % 500 == 0:
                print(f"\r  {i}/{len(rows)}", end="", flush=True)
        print()

        # manifest inside the bundle -- the notebook reads only this
        sio = io.StringIO()
        wr = csv.DictWriter(sio, fieldnames=list(w_rows[0].keys()))
        wr.writeheader()
        wr.writerows(w_rows)
        out.writestr("manifest.csv", sio.getvalue())

        out.writestr("README.txt",
                     "BovineInsight LSD training bundle\n"
                     f"images: {len(w_rows)} JPEG, max {SIZE}px, q{QUALITY}\n"
                     "manifest.csv: file,label,source,group_id,split,severity,sha256\n"
                     "label 1 = lesion (lumpy), 0 = normal\n"
                     "split: test is LOCKED -- evaluate once, at the very end\n"
                     "Sources: Mendeley w36hpf86j2 (CC BY 4.0), "
                     "Roboflow cattle_disease-detection (CC BY 4.0),\n"
                     "severity from Zenodo LumpySkinDisease_DataHub (CC BY 4.0)\n")

    for z in zips.values():
        z.close()

    size = OUT.stat().st_size
    print(f"\nwrote {OUT}")
    print(f"  images packed : {len(w_rows)}")
    print(f"  source bytes  : {n_bytes_in/1e6:9.1f} MB")
    print(f"  bundle size   : {size/1e6:9.1f} MB   "
          f"({100*size/max(1,n_bytes_in):.1f}% of source)")
    if failed:
        print(f"  !! FAILED {len(failed)}:")
        for p, e in failed[:5]:
            print(f"     {p[:70]}  {e}")

    print("\n  split composition:")
    for k in sorted(stats):
        print(f"    {k[0]:8} label={k[1]}  {stats[k]:5d}")


if __name__ == "__main__":
    main()
