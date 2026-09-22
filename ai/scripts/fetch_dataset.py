"""Download the Mendeley Lumpy Skin Images Dataset (CC BY 4.0) to ai/data/.

Resumable, and verifies the published SHA-256. Everything stays local.
Citation: Kumar, Sachin; Shastri, Sourabh (2022), "Lumpy Skin Images Dataset",
Mendeley Data, V1, doi: 10.17632/w36hpf86j2.1
"""
import hashlib, os, sys, time, urllib.request

URL = ("https://data.mendeley.com/public-files/datasets/w36hpf86j2/files/"
       "9bac7442-f3db-4ed8-b39c-b66c48e3677f/file_downloaded")
SHA = "8f683d7b70727d92427560bedb68ee0250e3b04165b2a570136bc4be61802331"
SIZE = 161739104
DEST = os.path.join("ai", "data", "lumpy_skin_dataset.zip")

os.makedirs(os.path.dirname(DEST), exist_ok=True)
have = os.path.getsize(DEST) if os.path.exists(DEST) else 0
if have >= SIZE:
    print(f"already have {have} bytes")
else:
    headers = {"User-Agent": "Mozilla/5.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"resuming from {have:,}")
    req = urllib.request.Request(URL, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(DEST, "ab" if have else "wb") as f:
        got = have
        while chunk := r.read(262144):
            f.write(chunk)
            got += len(chunk)
            el = time.time() - t0
            pct = 100 * got / SIZE
            rate = (got - have) / el / 1024 if el else 0
            print(f"\r{pct:5.1f}%  {got/1e6:6.1f}/{SIZE/1e6:.1f} MB  {rate:6.0f} kB/s", end="", flush=True)
    print()

print("verifying sha256...")
h = hashlib.sha256()
with open(DEST, "rb") as f:
    while b := f.read(1 << 20):
        h.update(b)
ok = h.hexdigest() == SHA
print("SHA-256:", h.hexdigest())
print("MATCH" if ok else "MISMATCH — file is corrupt, delete and re-run")
sys.exit(0 if ok else 1)
