"""Download LumpySkinDisease_DataHub from Zenodo (CC BY 4.0). Resumable.

Zenodo record 11003967. Public, no auth.
"""
import os, sys, time, urllib.request, zipfile

URL = "https://zenodo.org/records/11003967/files/LumpySkinDisease_DataHub.zip?download=1"
SIZE = 511_548_905
DEST = os.path.join("ai", "data", "zenodo_lsd_datahub.zip")

os.makedirs(os.path.dirname(DEST), exist_ok=True)
have = os.path.getsize(DEST) if os.path.exists(DEST) else 0

if have < SIZE:
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://zenodo.org/records/11003967",
    }   # Zenodo 403s the bare API URL / minimal UA -- browser headers required
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"resuming from {have:,}")
    req = urllib.request.Request(URL, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r, open(DEST, "ab" if have else "wb") as f:
        got = have
        while chunk := r.read(262144):
            f.write(chunk); got += len(chunk)
            el = time.time() - t0
            rate = (got - have) / el / 1024 if el else 0
            eta = (SIZE - got) / (rate * 1024) / 60 if rate else 0
            print(f"\r{100*got/SIZE:5.1f}%  {got/1e6:6.1f}/{SIZE/1e6:.1f} MB  "
                  f"{rate:6.0f} kB/s  ETA {eta:4.1f} min", end="", flush=True)
    print()
else:
    print(f"already complete: {have:,} bytes")

print("\nverifying zip integrity...")
try:
    with zipfile.ZipFile(DEST) as z:
        bad = z.testzip()
        names = z.namelist()
    if bad:
        print("CORRUPT member:", bad); sys.exit(1)
    print(f"OK - {len(names)} entries")
    print("\nTOP-LEVEL CONTENTS:")
    import collections
    c = collections.Counter(n.split("/")[0] for n in names)
    for k, v in c.most_common(30):
        print(f"  {v:6d}  {k}")
    imgs = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    print(f"\nIMAGE FILES: {len(imgs)}")
    dirs = collections.Counter("/".join(n.split("/")[:-1]) for n in imgs)
    print("IMAGE FOLDERS:")
    for k, v in dirs.most_common(20):
        print(f"  {v:6d}  {k}")
except zipfile.BadZipFile:
    print("BAD ZIP - delete and re-run"); sys.exit(1)
