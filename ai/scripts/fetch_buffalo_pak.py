"""Download Buffalo-Pak (Mendeley vdgnxsm692) -- Pakistani buffalo breed images.

WHY THIS MATTERS
----------------
Every image v1.0.0 trained on is CATTLE. Buffalo are ~59% of Pakistan's milk,
are black-hided, and LSD presents differently in them. "No buffalo validation"
is the largest open limitation in the model card.

Buffalo-Pak is 325 labelled Pakistani buffalo images (Nili-Ravi, Khundi, Mixed),
presumed healthy since it is a BREED classification dataset. That makes it a
free specificity test: run v1.0.0 over them and count the false positives.

Mendeley's CDN drops the connection partway, so this retries with HTTP Range
resume and only verifies once the byte count is exact.

Citation: Buffalo-Pak, Mendeley Data V2, doi:10.17632/vdgnxsm692.2
"""
import collections, os, sys, time, urllib.error, urllib.request, zipfile

URL = ("https://data.mendeley.com/public-files/datasets/vdgnxsm692/files/"
       "12298bd4-2825-4fc3-83f3-73b75837f811/file_downloaded")
SIZE = 268_861_655                      # confirmed via Content-Range
DEST = os.path.join("ai", "data", "buffalo_pak.zip")
MAX_TRIES = 40

BASE = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/131.0.0.0 Safari/537.36"),
        "Accept": "*/*",
        "Referer": "https://data.mendeley.com/datasets/vdgnxsm692/2"}

os.makedirs(os.path.dirname(DEST), exist_ok=True)

for attempt in range(1, MAX_TRIES + 1):
    have = os.path.getsize(DEST) if os.path.exists(DEST) else 0
    if have >= SIZE:
        break
    h = dict(BASE)
    h["Range"] = f"bytes={have}-"
    print(f"\n[try {attempt}] resuming at {have/1e6:.1f}/{SIZE/1e6:.1f} MB")
    try:
        t0, start = time.time(), have
        with urllib.request.urlopen(urllib.request.Request(URL, headers=h),
                                    timeout=120) as r, open(DEST, "ab") as f:
            got = have
            while chunk := r.read(262144):
                f.write(chunk); got += len(chunk)
                el = time.time() - t0
                rate = (got - start) / el / 1024 if el else 0
                eta = (SIZE - got) / (rate * 1024) / 60 if rate else 0
                print(f"\r  {100*got/SIZE:5.1f}%  {got/1e6:7.1f} MB  "
                      f"{rate:6.0f} kB/s  ETA {eta:4.1f} min", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n  dropped: {type(e).__name__} {e} -- retrying")
        time.sleep(3)

have = os.path.getsize(DEST)
if have < SIZE:
    sys.exit(f"INCOMPLETE after {MAX_TRIES} tries: {have}/{SIZE}")

print(f"\ncomplete: {have:,} bytes")
print("verifying...")
with zipfile.ZipFile(DEST) as z:
    bad = z.testzip()
    if bad:
        sys.exit(f"CORRUPT member: {bad}")
    names = z.namelist()
imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"OK - {len(names)} entries, {len(imgs)} images")
print("FOLDERS:")
for k, v in collections.Counter(
        "/".join(n.split("/")[:-1]) for n in imgs).most_common(25):
    print(f"  {v:6d}  {k}")
