"""Step 3 — Group-aware, source-stratified splits.

Reads  ai/data/manifest.csv   (+ ai/data/severity_labels.csv if present)
Writes ai/data/splits.csv     group_id, split  ->  "test" | "fold0".."fold4"
       ai/data/train_manifest.csv   manifest + split + severity, ready for Kaggle
       ai/reports/splits.md

WHY NOT sklearn.StratifiedGroupKFold
------------------------------------
It stratifies on `y` only. Our problem needs stratification on the JOINT
(label, source), because source almost determines label in this data:

    mendeley   P(lesion|source) = 0.299
    roboflow   P(lesion|source) = 0.826

If a fold ends up source-skewed, a model can score well by recognising WHICH
DATASET an image came from -- camera, resolution, compression, framing -- and
never look at a lesion. Balancing all four (label x source) cells across every
fold removes the easy version of that shortcut and, more importantly, lets us
measure per-source performance honestly.

Implemented directly in numpy: no new dependency, deterministic, auditable.

TWO HARD RULES (medical-imaging failure literature)
---------------------------------------------------
1. An entire duplicate GROUP goes to exactly one split. Never divided.
2. The test set is carved out FIRST and touched exactly once, at the very end.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ai" / "data"
REPORTS = ROOT / "ai" / "reports"

SEED = 42
N_FOLDS = 5
TEST_FRAC = 0.20


def load_manifest():
    rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
    if not rows:
        raise SystemExit("manifest.csv is empty -- run dedupe.py first")
    return rows


def load_severity():
    f = DATA / "severity_labels.csv"
    if not f.exists():
        return {}
    return {r["sha256"]: r["severity"]
            for r in csv.DictReader(open(f, encoding="utf-8"))}


def build_groups(rows):
    """group_id -> dict(label, source, n_images, severities)."""
    g = {}
    for r in rows:
        gid = r["group_id"]
        if gid not in g:
            g[gid] = {"label": int(r["label"]), "source": r["source"],
                      "n": 0, "sev": Counter()}
        g[gid]["n"] += 1
        # a group's source is the one that contributed most of its images
        g[gid].setdefault("srcs", Counter())[r["source"]] += 1
    for gid, d in g.items():
        d["source"] = d["srcs"].most_common(1)[0][0]
        del d["srcs"]
    return g


def stratified_group_assign(groups, n_bins, seed, bin_names):
    """Greedy: distribute each (label, source) stratum evenly over bins.

    Within a stratum, groups are shuffled (seeded) then handed to whichever bin
    currently holds the fewest groups OF THAT STRATUM. Ties broken by total size,
    so bins stay balanced in image count too.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for gid, d in groups.items():
        strata[(d["label"], d["source"])].append(gid)

    assign = {}
    per_bin_stratum = defaultdict(Counter)   # bin -> stratum -> count
    per_bin_images = Counter()

    for stratum in sorted(strata):
        gids = sorted(strata[stratum])       # sort first => deterministic
        rng.shuffle(gids)
        for gid in gids:
            b = min(bin_names,
                    key=lambda b: (per_bin_stratum[b][stratum], per_bin_images[b]))
            assign[gid] = b
            per_bin_stratum[b][stratum] += 1
            per_bin_images[b] += groups[gid]["n"]
    return assign


def summarise(groups, assign, bins):
    out = {}
    for b in bins:
        gids = [g for g, x in assign.items() if x == b]
        lab = Counter(groups[g]["label"] for g in gids)
        src = Counter(groups[g]["source"] for g in gids)
        imgs = sum(groups[g]["n"] for g in gids)
        out[b] = {"groups": len(gids), "images": imgs,
                  "lesion": lab[1], "normal": lab[0],
                  "p_lesion": lab[1] / max(1, len(gids)), "src": dict(src)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    rows = load_manifest()
    sev = load_severity()
    groups = build_groups(rows)
    print(f"loaded {len(rows)} images in {len(groups)} groups")

    # ---------------------------------------------- 1. carve out the test set
    # Done FIRST, before anything else. 5 bins -> take 1 as test (=20%).
    n_test_bins = max(1, round(1 / TEST_FRAC))
    tmp_bins = [f"b{i}" for i in range(n_test_bins)]
    tmp = stratified_group_assign(groups, n_test_bins, args.seed, tmp_bins)

    test_gids = {g for g, b in tmp.items() if b == "b0"}
    dev_gids = set(groups) - test_gids
    print(f"\nTEST  {len(test_gids)} groups   (locked -- touch ONCE, at the end)")
    print(f"DEV   {len(dev_gids)} groups   (all CV happens inside this)")

    # ---------------------------------------------- 2. CV folds inside dev
    dev_groups = {g: groups[g] for g in dev_gids}
    fold_names = [f"fold{i}" for i in range(args.folds)]
    fold_assign = stratified_group_assign(dev_groups, args.folds,
                                          args.seed + 1, fold_names)

    assign = {g: "test" for g in test_gids}
    assign.update(fold_assign)

    # ---------------------------------------------- 3. VERIFY (hard assertions)
    assert set(assign) == set(groups), "some group was not assigned"
    assert len(test_gids & dev_gids) == 0, "GROUP LEAK: test/dev overlap"
    for f in fold_names:
        fg = {g for g, b in assign.items() if b == f}
        assert not (fg & test_gids), f"GROUP LEAK: {f} overlaps test"
    seen = Counter(assign.values())
    assert sum(seen.values()) == len(groups)
    print("\n[PASS] every group assigned to exactly one split")
    print("[PASS] no group appears in both test and any fold")

    # ---------------------------------------------- 4. report
    bins = ["test"] + fold_names
    summ = summarise(groups, assign, bins)

    print(f"\n{'split':8} {'groups':>7} {'images':>7} {'lesion':>7} {'normal':>7} "
          f"{'P(les)':>7}  sources")
    for b in bins:
        s = summ[b]
        srcs = " ".join(f"{k}={v}" for k, v in sorted(s["src"].items()))
        print(f"{b:8} {s['groups']:7d} {s['images']:7d} {s['lesion']:7d} "
              f"{s['normal']:7d} {s['p_lesion']:7.3f}  {srcs}")

    ps = [summ[b]["p_lesion"] for b in bins]
    print(f"\nP(lesion) spread across splits: {min(ps):.3f} - {max(ps):.3f} "
          f"(range {max(ps) - min(ps):.3f})")
    if max(ps) - min(ps) > 0.05:
        print("  !! folds are label-imbalanced -- investigate")
    else:
        print("  folds are well balanced on label")

    # ---------------------------------------------- 5. write splits
    with open(DATA / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "split", "label", "source", "n_images"])
        for g in sorted(assign, key=lambda x: int(x)):
            d = groups[g]
            w.writerow([g, assign[g], d["label"], d["source"], d["n"]])
    print(f"\nwrote {DATA / 'splits.csv'}")

    # ---------------------------------------------- 6. training manifest
    n_sev = 0
    with open(DATA / "train_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "group_id", "split",
                    "severity", "sha256"])
        for r in rows:
            s = sev.get(r["sha256"], "")
            n_sev += bool(s)
            w.writerow([r["path"], r["label"], r["source"], r["group_id"],
                        assign[r["group_id"]], s, r["sha256"]])
    print(f"wrote {DATA / 'train_manifest.csv'}  "
          f"({len(rows)} rows, {n_sev} with severity)")

    # severity coverage per split -- matters for the Mild-recall test
    sevsplit = defaultdict(Counter)
    for r in rows:
        s = sev.get(r["sha256"])
        if s:
            sevsplit[assign[r["group_id"]]][s] += 1
    print("\nseverity coverage by split (for the Mild-vs-Severe recall test):")
    for b in bins:
        c = sevsplit[b]
        print(f"  {b:8} Normal={c['Normal']:4d}  Mild={c['Mild']:4d}  "
              f"Severe={c['Severe']:4d}")

    # ---------------------------------------------- 7. markdown report
    tab = "\n".join(
        f"| {b} | {summ[b]['groups']} | {summ[b]['images']} | {summ[b]['lesion']} | "
        f"{summ[b]['normal']} | {summ[b]['p_lesion']:.3f} | "
        + " ".join(f"{k}={v}" for k, v in sorted(summ[b]['src'].items())) + " |"
        for b in bins)
    sevtab = "\n".join(
        f"| {b} | {sevsplit[b]['Normal']} | {sevsplit[b]['Mild']} | "
        f"{sevsplit[b]['Severe']} |" for b in bins)

    (REPORTS / "splits.md").write_text(f"""# Step 3 — Split Report

**Seed:** {args.seed} · **Folds:** {args.folds} · **Test fraction:** {TEST_FRAC:.0%}

Stratified on the **joint (label, source)**, not label alone — because source
nearly determines label in this data (`P(lesion|mendeley)=0.299`,
`P(lesion|roboflow)=0.826`). Source-skewed folds would let a model score well by
recognising which dataset an image came from rather than by finding lesions.

## Composition

| Split | Groups | Images | Lesion | Normal | P(lesion) | Sources |
|---|---|---|---|---|---|---|
{tab}

P(lesion) spread across splits: **{min(ps):.3f} – {max(ps):.3f}**

## Severity coverage (Zenodo labels, joined by SHA-256)

| Split | Normal | Mild | Severe |
|---|---|---|---|
{sevtab}

**Mild recall is the number that matters for an early-detection claim.** A model
that only catches Severe cases is detecting what the farmer can already see.

## Verification

- [PASS] every group assigned to exactly one split
- [PASS] no group appears in both test and any fold
- Test set is **locked**. Touch it once, at the very end. Tuning against it
  makes it a validation set, and you will need a fresh one.

## Still to do in training

Per-source metrics must be reported separately. Consider a source-adversarial
check: train a classifier to predict SOURCE from the image. If it scores near
100%, the shortcut is trivially available and Grad-CAM becomes the deciding test.
""", encoding="utf-8")
    print(f"wrote {REPORTS / 'splits.md'}")


if __name__ == "__main__":
    main()
