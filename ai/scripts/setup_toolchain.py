"""Headless Flutter + Android toolchain install, then build the APK.

Everything goes under C:\src. No GUI, no Android Studio.
Resumable: re-run if a download drops.
"""
import os, subprocess, sys, time, urllib.request, zipfile, shutil

SRC = r"C:\src"
DL = os.path.join(SRC, "_dl")
os.makedirs(DL, exist_ok=True)

ITEMS = [
    ("jdk17.zip",
     "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse",
     os.path.join(SRC, "jdk")),
    ("cmdline-tools.zip",
     "https://dl.google.com/android/repository/commandlinetools-win-15859902_latest.zip",
     os.path.join(SRC, "android-sdk", "cmdline-tools")),
    ("flutter.zip",
     # Flutter 3.24.3 == Dart 3.5.3, matching this project's pubspec
     "https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.24.3-stable.zip",
     SRC),
]

H = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")}


def size_of(url):
    h = dict(H); h["Range"] = "bytes=0-0"
    r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60)
    cr = r.headers.get("Content-Range")
    return int(cr.split("/")[-1]) if cr else None


def download(name, url):
    dest = os.path.join(DL, name)
    total = size_of(url)
    for attempt in range(1, 30):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if total and have >= total:
            print(f"  {name}: complete ({have/1e6:.0f} MB)")
            return dest
        h = dict(H)
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            t0, start = time.time(), have
            with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                        timeout=120) as r, open(dest, "ab") as f:
                got = have
                while chunk := r.read(262144):
                    f.write(chunk); got += len(chunk)
                    el = time.time() - t0
                    rate = (got - start) / el / 1024 if el else 0
                    pct = f"{100*got/total:5.1f}%" if total else "  ?  "
                    print(f"\r  {name} {pct} {got/1e6:7.1f} MB {rate:6.0f} kB/s",
                          end="", flush=True)
            print()
            if not total:
                return dest
        except Exception as e:
            print(f"\n  {name}: {type(e).__name__} - retry {attempt}")
            time.sleep(3)
    return dest


print("=== DOWNLOADING TOOLCHAIN ===")
paths = {}
for name, url, _ in ITEMS:
    print(f"\n{name}")
    paths[name] = download(name, url)

print("\n=== EXTRACTING ===")
for name, url, target in ITEMS:
    z = paths[name]
    marker = {"jdk17.zip": os.path.join(SRC, "jdk"),
              "cmdline-tools.zip": os.path.join(SRC, "android-sdk", "cmdline-tools", "latest"),
              "flutter.zip": os.path.join(SRC, "flutter")}[name]
    if os.path.exists(marker) and os.listdir(marker):
        print(f"  {name}: already extracted")
        continue
    print(f"  extracting {name} ...")
    tmp = os.path.join(DL, name + ".x")
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(tmp)
    entries = os.listdir(tmp)
    if name == "jdk17.zip":
        inner = os.path.join(tmp, entries[0])
        os.makedirs(SRC, exist_ok=True)
        shutil.move(inner, os.path.join(SRC, "jdk"))
    elif name == "cmdline-tools.zip":
        os.makedirs(os.path.join(SRC, "android-sdk", "cmdline-tools"), exist_ok=True)
        shutil.move(os.path.join(tmp, "cmdline-tools"),
                    os.path.join(SRC, "android-sdk", "cmdline-tools", "latest"))
    else:
        shutil.move(os.path.join(tmp, "flutter"), os.path.join(SRC, "flutter"))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {name}: done")

print("\n=== INSTALLED ===")
for p in [os.path.join(SRC, "jdk", "bin", "java.exe"),
          os.path.join(SRC, "flutter", "bin", "flutter.bat"),
          os.path.join(SRC, "android-sdk", "cmdline-tools", "latest", "bin",
                       "sdkmanager.bat")]:
    print(("  OK   " if os.path.exists(p) else "  MISS ") + p)
