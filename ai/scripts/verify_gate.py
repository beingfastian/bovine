"""Local verification of the two-output (lesion + species) ONNX artifact.

The Kaggle kernel measured the species head on its held-out split and wrote
ai/reports/gate_report.json. This script asks the questions that only make
sense against the FILE THAT SHIPS, on this disk:

  1. Did the lesion output move? It must not. Same layers, same weights --
     the new fp32 must match the verified fp32 to float precision, and the new
     fp16 must reproduce the model card's confusion matrix exactly.
  2. What does the artifact say about the local buffalo photographs and the
     cattle test split? (The buffalo set was 75% training data for the head,
     so its number here is a sanity check, not an accuracy claim -- the honest
     held-out figure is in gate_report.json.)
  3. Which otherMax to ship: the tightest gate that still keeps >= 99% of real
     cattle AND >= 99% of real buffalo on the kernel's held-out split. A farm
     photo wrongly turned away is a lost data point, which for this app is the
     worst outcome there is.
  4. Sanity on real non-animal photographs, if any exist on this machine.

Run:  python ai/scripts/verify_gate.py
"""
from __future__ import annotations

import csv
import glob
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
import os
# Override with GATE_OUT / GATE_TAG to verify another run's artifact, and
# GATE_OOD_MAX / GATE_OTHER_MAX to CHECK fixed thresholds instead of choosing.
OUT_DIR = Path(os.environ.get("GATE_OUT", ROOT / "ai/kernels/bovine-lsd-gate/out"))
TAG = os.environ.get("GATE_TAG", "bi-lsd-mnv3l-v1.0.0-gate")
NEW32 = OUT_DIR / f"{TAG}.onnx"
NEW16 = OUT_DIR / f"{TAG}.fp16.onnx"
FIXED_OOD = float(os.environ["GATE_OOD_MAX"]) if "GATE_OOD_MAX" in os.environ else None
FIXED_T = float(os.environ["GATE_OTHER_MAX"]) if "GATE_OTHER_MAX" in os.environ else None
OLD32 = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.onnx"
OLD16 = ROOT / "ai/models/bi-lsd-mnv3l-v1.0.0/bi-lsd-mnv3l-v1.0.0.fp16.onnx"
GATE_REPORT = OUT_DIR / "gate_report.json"
BUNDLE = ROOT / "ai/data/bovine_lsd_bundle.zip"
BUFFALO = ROOT / "ai/data/buffalo_pak.zip"
REPORT = ROOT / "ai/reports/gate_verification.json"
TAU = 0.50
KEEP_MIN = 0.99
CLASSES = ["cattle", "buffalo", "other"]


def preprocess(im: Image.Image) -> np.ndarray:
    """RGB, EXIF-upright, bilinear 224, RAW [0,255]. Mirrors web/lib/preprocess.ts."""
    im = ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, np.float32)[None, ...]


class Model:
    def __init__(self, path: Path):
        m = onnx.load(str(path))
        onnx.checker.check_model(m)
        ops = {n.op_type for n in m.graph.node}
        banned = {o for o in ops if "Random" in o or "Stateless" in o or o in ("If", "Loop")}
        assert not banned, f"{path.name}: training-time ops in graph: {banned}"
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.names = [o.name for o in self.sess.get_outputs()]
        shapes = {o.name: list(o.shape) for o in self.sess.get_outputs()}
        # Outputs by NAME. The gate artifact has two [1,1] outputs (lesion and
        # ood), so shape alone cannot tell them apart. The single-output
        # shipped artifact is recognised by having exactly one output.
        if len(self.names) == 1:
            self.k_les, self.k_spe, self.k_ood = self.names[0], None, None
        else:
            assert {"lesion", "species", "ood"} <= set(self.names), f"{path.name}: outputs {self.names}"
            assert shapes["lesion"] == [1, 1] and shapes["species"] == [1, 3] and shapes["ood"] == [1, 1], shapes
            self.k_les, self.k_spe, self.k_ood = "lesion", "species", "ood"
        self.path = path

    def run(self, x: np.ndarray):
        res = dict(zip(self.names, self.sess.run(None, {self.inp: x})))
        les = float(res[self.k_les].ravel()[0])
        spe = res[self.k_spe].ravel().astype(np.float64) if self.k_spe else None
        ood = float(res[self.k_ood].ravel()[0]) if self.k_ood else None
        return les, spe, ood


def metrics(y, p, tau=TAU):
    yh = (p >= tau).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    return {"balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    for p in (NEW32, NEW16, OLD32, OLD16):
        assert p.exists(), f"missing {p.relative_to(ROOT)}"
    new32, new16, old32, old16 = Model(NEW32), Model(NEW16), Model(OLD32), Model(OLD16)
    assert new32.k_spe and new16.k_spe and new32.k_ood and new16.k_ood, "new model lacks species/ood outputs"
    report: dict = {"artifacts": {
        "new_fp32": {"file": NEW32.name, "bytes": NEW32.stat().st_size},
        "new_fp16": {"file": NEW16.name, "bytes": NEW16.stat().st_size},
        "shipped_fp16": {"file": OLD16.name, "bytes": OLD16.stat().st_size},
    }}
    print(f"new fp16: {NEW16.stat().st_size/1e6:.2f} MB   shipped fp16: {OLD16.stat().st_size/1e6:.2f} MB")

    # ------------------------------------------------------- 1. lesion parity
    z = zipfile.ZipFile(BUNDLE)
    rows = [r for r in csv.DictReader(io.StringIO(z.read("manifest.csv").decode()))
            if r["split"] == "test"]
    y = np.array([int(r["label"]) for r in rows])
    print(f"\nscoring {len(rows)} locked cattle test images through 4 models…")
    L = {k: [] for k in ("new32", "new16", "old32", "old16")}
    S16, S32, D16, D32 = [], [], [], []
    for k, r in enumerate(rows, 1):
        x = preprocess(Image.open(io.BytesIO(z.read(r["file"]))))
        a, s32, d32_ = new32.run(x); L["new32"].append(a); S32.append(s32); D32.append(d32_)
        b, s16, d16_ = new16.run(x); L["new16"].append(b); S16.append(s16); D16.append(d16_)
        L["old32"].append(old32.run(x)[0]); L["old16"].append(old16.run(x)[0])
        if k % 100 == 0 or k == len(rows):
            print(f"\r  {k}/{len(rows)}", end="", flush=True)
    print()
    L = {k: np.array(v) for k, v in L.items()}
    S16, S32, D16, D32 = np.array(S16), np.array(S32), np.array(D16), np.array(D32)

    d32 = float(np.max(np.abs(L["new32"] - L["old32"])))
    d16 = float(np.max(np.abs(L["new16"] - L["old16"])))
    agree16 = float(((L["new16"] >= TAU) == (L["old16"] >= TAU)).mean())
    m_new16, m_old16 = metrics(y, L["new16"]), metrics(y, L["old16"])
    print("\n" + "=" * 66)
    print("1. LESION OUTPUT — did it move?")
    print("=" * 66)
    print(f"  new fp32 vs verified fp32   max abs diff {d32:.2e}")
    print(f"  new fp16 vs shipped fp16    max abs diff {d16:.4f}   decision agreement {agree16:.4f}")
    print(f"  new fp16 confusion  TP={m_new16['tp']} TN={m_new16['tn']} FP={m_new16['fp']} FN={m_new16['fn']}  "
          f"bal.acc {m_new16['balanced_accuracy']:.4f}")
    print(f"  model card          TP=389 TN=177 FP=30 FN=15  bal.acc 0.9090")
    report["lesion_parity"] = {"new32_vs_old32_max_abs_diff": d32, "new16_vs_old16_max_abs_diff": d16,
                               "new16_vs_old16_decision_agreement": agree16,
                               "new16_metrics": m_new16, "old16_metrics": m_old16}

    cattle_pred = S16.argmax(1)
    report["cattle_test_species"] = {
        "detected_cattle": float((cattle_pred == 0).mean()),
        "detected_buffalo": float((cattle_pred == 1).mean()),
        "detected_other": float((cattle_pred == 2).mean()),
        "p_other_p50": float(np.percentile(S16[:, 2], 50)),
        "p_other_p99": float(np.percentile(S16[:, 2], 99)),
        "fp16_vs_fp32_argmax_agreement": float((S16.argmax(1) == S32.argmax(1)).mean()),
    }
    print(f"\n  cattle test as species: cattle {report['cattle_test_species']['detected_cattle']:.4f} "
          f"buffalo {report['cattle_test_species']['detected_buffalo']:.4f} "
          f"other {report['cattle_test_species']['detected_other']:.4f}   "
          f"(fp16/fp32 argmax agree {report['cattle_test_species']['fp16_vs_fp32_argmax_agreement']:.4f})")

    ood_rel = float(np.max(np.abs(D16 - D32) / (D32 + 1e-6)))
    report["cattle_test_ood"] = {"d2_p50": float(np.percentile(D32, 50)), "d2_p99": float(np.percentile(D32, 99)),
                                 "d2_p995": float(np.percentile(D32, 99.5)), "d2_max": float(D32.max()),
                                 "fp16_vs_fp32_max_rel_err": ood_rel}
    print(f"  cattle test ood d2: p50 {np.percentile(D32,50):.1f}  p99 {np.percentile(D32,99):.1f}  "
          f"p99.5 {np.percentile(D32,99.5):.1f}  max {D32.max():.1f}   (fp16 vs fp32 rel err {ood_rel:.2e})")

    # ---------------------------------------------------- 2. local buffalo set
    bz = zipfile.ZipFile(BUFFALO)
    bnames = [n for n in bz.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nscoring {len(bnames)} Buffalo-Pak photographs (full-res, EXIF-upright)…")
    B, Bl, Bd, breeds = [], [], [], []
    for k, n in enumerate(bnames, 1):
        x = preprocess(Image.open(io.BytesIO(bz.read(n))))
        les, spe, d2_ = new16.run(x); B.append(spe); Bl.append(les); Bd.append(d2_); breeds.append(n.split("/")[1])
        if k % 50 == 0 or k == len(bnames):
            print(f"\r  {k}/{len(bnames)}", end="", flush=True)
    print()
    B, Bl, Bd, breeds = np.array(B), np.array(Bl), np.array(Bd), np.array(breeds)
    bpred = B.argmax(1)
    per_breed = {b: {"n": int((breeds == b).sum()),
                     "detected_buffalo": float((bpred[breeds == b] == 1).mean())}
                 for b in sorted(set(breeds))}
    print("=" * 66)
    print("2. BUFFALO-PAK (caveat: ~75% of these trained the head)")
    print("=" * 66)
    print(f"  detected buffalo {float((bpred == 1).mean()):.4f}   cattle {float((bpred == 0).mean()):.4f}   "
          f"other {float((bpred == 2).mean()):.4f}")
    for b, v in per_breed.items():
        print(f"     {b:10} n={v['n']:3d}  buffalo {v['detected_buffalo']:.4f}")
    print(f"  lesion false-alarm rate at tau (card: 0.1235): {float((Bl >= TAU).mean()):.4f}")
    print(f"  ood d2: p50 {np.percentile(Bd,50):.1f}  p99 {np.percentile(Bd,99):.1f}  max {Bd.max():.1f}")
    report["buffalo_pak_ood"] = {"d2_p50": float(np.percentile(Bd, 50)), "d2_p99": float(np.percentile(Bd, 99)),
                                 "d2_max": float(Bd.max())}
    report["buffalo_pak_species"] = {"caveat": "includes the head's training images; see gate_report.json for held-out",
                                     "detected_buffalo": float((bpred == 1).mean()),
                                     "detected_cattle": float((bpred == 0).mean()),
                                     "detected_other": float((bpred == 2).mean()),
                                     "per_breed": per_breed,
                                     "lesion_false_alarm_rate": float((Bl >= TAU).mean())}

    # ------------------------------------------------------ 3. choose otherMax
    print("\n" + "=" * 66)
    print(f"3. GATE THRESHOLD  keep >= {KEEP_MIN:.2f} of every real animal")
    print("=" * 66)
    gate = json.loads(GATE_REPORT.read_text(encoding="utf-8")) if GATE_REPORT.exists() else {}
    ood_sweep = (gate.get("ood") or {}).get("sweep")
    chosen_ood = None
    if FIXED_OOD is not None:
        # Thresholds were chosen elsewhere (on validation data). Check, do not choose.
        gate, ood_sweep = {}, None
        kc = float((D16 < FIXED_OOD).mean()); kb = float((Bd < FIXED_OOD).mean())
        report["ood_gate"] = {"recommendedOodMax": FIXED_OOD, "source": "fixed (GATE_OOD_MAX)",
                              "local_cattle_test_kept": kc, "local_buffalo_kept": kb}
        print(f"  FIXED oodMax {FIXED_OOD:.2f}: local cattle test kept {kc:.4f}, local buffalo kept {kb:.4f}")
    if not ood_sweep and gate:
        # The kernel's D values only reached stdout. Rebuild the sweep the way
        # the kernel defined it -- D = max(pct(cattle test d2), pct(buffalo d2))
        # at each keep level -- from the d2 this artifact gives the local
        # images. Cattle test is fully held-out. The buffalo set includes the
        # 75% the head trained on, which makes its percentile slightly LOW and
        # the gate therefore slightly STRICT for buffalo; the honest buffalo
        # figure at the chosen D is printed below and checked >= 0.99.
        # Novel rejection comes from the kernel's fp16 pass (progress.txt).
        novel = ((gate.get("verification") or {}).get("novel_rejected_fp16_combined") or {})
        ood_sweep = []
        for keep in (0.98, 0.99, 0.995, 0.999):
            D = float(max(np.percentile(D16, 100 * keep), np.percentile(Bd, 100 * keep)))
            ood_sweep.append({"keep_target": keep, "D": D,
                              "cattle_kept": float((D16 < D).mean()), "buffalo_kept": float((Bd < D).mean()),
                              "other_test_rejected": float("nan"),
                              "novel_rejected": novel.get(f"keep{keep}_T0.7", float("nan")),
                              "novel_by_source": {}, "D_source": "local (cattle test + buffalo, fp16 artifact)"})
        print("  (D recomputed locally from the fp16 artifact; novel rejection from the kernel's fp16 pass, combined with T=0.7)")
    if ood_sweep:
        print("  OOD half (held-out + NOVEL, from kernel):")
        print(f"  {'keep':>6} {'D':>8} {'cattle kept':>12} {'buffalo kept':>13} {'other-test rej':>15} {'NOVEL rej':>10}")
        for row in ood_sweep:
            ok = row["cattle_kept"] >= KEEP_MIN and row["buffalo_kept"] >= KEEP_MIN
            mark = "  <- tightest that qualifies" if ok and chosen_ood is None else ""
            if ok and chosen_ood is None:
                chosen_ood = row
            print(f"  {row['keep_target']:6.3f} {row['D']:8.1f} {row['cattle_kept']:12.4f} {row['buffalo_kept']:13.4f} "
                  f"{row['other_test_rejected']:15.4f} {row['novel_rejected']:10.4f}{mark}")
        if chosen_ood:
            print("     novel rejection by source:", {k: round(v, 3) for k, v in chosen_ood["novel_by_source"].items()})
            local_ood_kept = float((D16 < chosen_ood["D"]).mean())
            local_buf_kept = float((Bd < chosen_ood["D"]).mean())
            print(f"     at D={chosen_ood['D']:.1f}: local cattle test kept {local_ood_kept:.4f}, "
                  f"local buffalo kept {local_buf_kept:.4f}")
            report["ood_gate"] = {"recommendedOodMax": chosen_ood["D"], "at_threshold": chosen_ood,
                                  "local_cattle_test_kept": local_ood_kept, "local_buffalo_kept": local_buf_kept}
    sweep = None if FIXED_T is not None else ((gate.get("head") or {}).get("softmax_only_sweep") or gate.get("gate_sweep"))
    chosen = {"T": FIXED_T, "cattle_kept": float((S16[:, 2] < FIXED_T).mean()),
              "buffalo_kept": float((B[:, 2] < FIXED_T).mean()), "other_rejected": None} if FIXED_T is not None else None
    if sweep:
        print("\n  softmax P(other) half:")
        print("  (held-out numbers from the kernel)")
        print(f"  {'T':>5} {'cattle kept':>12} {'buffalo kept':>13} {'other rejected':>15}")
        for row in sorted(sweep, key=lambda r: r["T"]):
            row.setdefault("other_rejected", row.get("other_test_rejected"))
            ok = row["cattle_kept"] >= KEEP_MIN and row["buffalo_kept"] >= KEEP_MIN
            mark = "  <- tightest that qualifies" if ok and chosen is None else ""
            if ok and chosen is None:
                chosen = row
            print(f"  {row['T']:5.2f} {row['cattle_kept']:12.4f} {row['buffalo_kept']:13.4f} "
                  f"{row['other_rejected']:15.4f}   novel {row.get('novel_rejected', float('nan')):.4f}{mark}")
        if chosen is None:
            chosen = max(sweep, key=lambda r: r["T"])
            print(f"  no threshold keeps >= {KEEP_MIN} of both species; falling back to loosest T={chosen['T']}")
    elif chosen is None:
        # No kernel report on disk. Choose PROVISIONALLY from what is clean
        # locally: the cattle test split (never seen by the head) and the
        # buffalo set (caveated). No non-animal held-out here, so the rejection
        # rate is unknown until the kernel report lands.
        print("  (no gate_report.json -- provisional choice from local animals only;")
        print("   non-animal rejection rate UNKNOWN until the kernel report lands)")
        print(f"  {'T':>5} {'cattle kept':>12} {'buffalo kept':>13}")
        for T in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            ck = float((S16[:, 2] < T).mean()); bk = float((B[:, 2] < T).mean())
            ok = ck >= KEEP_MIN and bk >= KEEP_MIN
            mark = "  <- tightest that qualifies" if ok and chosen is None else ""
            row = {"T": T, "cattle_kept": ck, "buffalo_kept": bk, "other_rejected": None}
            if ok and chosen is None:
                chosen = row
            print(f"  {T:5.2f} {ck:12.4f} {bk:13.4f}{mark}")
        if chosen is None:
            chosen = {"T": 0.95, "cattle_kept": float((S16[:, 2] < 0.95).mean()),
                      "buffalo_kept": float((B[:, 2] < 0.95).mean()), "other_rejected": None}
    # local check of the same T on the cattle test split
    Dfix = (report.get("ood_gate") or {}).get("recommendedOodMax", float("inf"))
    joint_c = ~((D16 >= Dfix) | (S16[:, 2] >= chosen["T"]))
    joint_b = ~((Bd >= Dfix) | (B[:, 2] >= chosen["T"]))
    report["joint_retention_local"] = {"cattle_test": float(joint_c.mean()),
                                       "lesion_positive_cattle_test": float(joint_c[y == 1].mean()),
                                       "buffalo_pak": float(joint_b.mean())}
    print(f"  JOINT retention (both halves): cattle {joint_c.mean():.4f}  "
          f"lesion-positive cattle {joint_c[y == 1].mean():.4f}  buffalo {joint_b.mean():.4f}")
    local_kept = float(joint_c.mean())
    rej_txt = (f"rejects {chosen['other_rejected']:.1%} of held-out non-animals"
               if chosen.get("other_rejected") is not None else "non-animal rejection rate not yet measured")
    print(f"\n  recommended otherMax = {chosen['T']}   ({rej_txt}; "
          f"keeps {local_kept:.4f} of the local cattle test split)")
    report["gate"] = {"keep_min": KEEP_MIN, "recommendedOtherMax": chosen["T"],
                      "at_threshold": chosen, "local_cattle_test_kept": local_kept,
                      "kernel_heldout_accuracy": gate.get("heldout_accuracy"),
                      "kernel_species": gate.get("species"),
                      "lesion_model_on_nonanimals_before_gate": gate.get("lesion_model_on_nonanimals")}

    # -------------------------------------------------- 4. non-animals, local
    wallpapers = sorted(glob.glob(r"C:\Windows\Web\Wallpaper\*\*.jpg"))[:12]
    if wallpapers:
        print("\n" + "=" * 66)
        print("4. NON-ANIMAL SANITY  (Windows wallpapers on this machine)")
        print("=" * 66)
        rej = 0
        rows_w = []
        Dmax = (report.get("ood_gate") or {}).get("recommendedOodMax", float("inf"))
        for p in wallpapers:
            les, spe, d2_ = new16.run(preprocess(Image.open(p)))
            by = [h for h, hit in (("ood", d2_ >= Dmax), ("softmax", spe[2] >= chosen["T"])) if hit]
            r = bool(by); rej += r
            rows_w.append({"file": Path(p).name, "p_other": float(spe[2]), "d2": d2_, "lesion_p": les,
                           "rejected": r, "by": by})
            print(f"  {Path(p).name:12} p_other {spe[2]:.3f}  d2 {d2_:8.1f}  lesion_p {les:.3f}  "
                  f"{'REJECTED by ' + '+'.join(by) if r else 'passed (!)'}")
        print(f"  rejected {rej}/{len(wallpapers)}")
        report["nonanimal_local"] = rows_w

    # ------------------------------------------------------------ verdict
    checks = [
        ("lesion fp32 unchanged (max abs diff < 1e-5)", d32 < 1e-5),
        ("lesion fp16 decision agreement with shipped == 1.0", agree16 == 1.0),
        ("lesion fp16 confusion matrix == model card",
         (m_new16["tp"], m_new16["tn"], m_new16["fp"], m_new16["fn"]) == (389, 177, 30, 15)),
        ("cattle test split kept by gate >= 0.99", local_kept >= KEEP_MIN),
        ("species fp16/fp32 argmax agreement >= 0.99",
         report["cattle_test_species"]["fp16_vs_fp32_argmax_agreement"] >= 0.99),
        ("ood fp16/fp32 max rel err < 2%", report["cattle_test_ood"]["fp16_vs_fp32_max_rel_err"] < 0.02),
        ("ood threshold chosen from a held-out sweep or fixed by the kernel", "ood_gate" in report),
        ("lesion-positive cattle kept by the joint gate >= 0.99",
         report.get("joint_retention_local", {}).get("lesion_positive_cattle_test", 0) >= KEEP_MIN),
        ("local buffalo kept by ood gate >= 0.99",
         (report.get("ood_gate") or {}).get("local_buffalo_kept", 0) >= KEEP_MIN),
        ("real non-animal photos on this machine: >= 90% rejected",
         (not wallpapers) or (sum(r["rejected"] for r in report.get("nonanimal_local", [])) >= 0.9 * len(wallpapers))),
    ]
    print("\n" + "=" * 66)
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    report["allChecksPassed"] = all(c[1] for c in checks)
    print("\nOVERALL:", "PASS" if report["allChecksPassed"] else "FAIL")
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
