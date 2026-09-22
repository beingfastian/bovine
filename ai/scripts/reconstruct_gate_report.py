"""Rebuild gate_report.json from the kernel's progress.txt.

Kernel v5 computed everything and then failed on its final json.dump (a name
collision between the sweep list and the Keras model, fixed in v6). The
artifacts it produced were fully verified by its cell 11, and the numbers were
written line by line to progress.txt as they were computed. This turns those
lines back into data so nothing rests on a screenshot of a log.

Fields that only ever went to stdout (the per-source novel breakdown and the
softmax-only sweep) are marked absent rather than guessed.

Run:  python ai/scripts/reconstruct_gate_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = ROOT / "ai/kernels/bovine-lsd-gate/out/progress.txt"
OUT = ROOT / "ai/reports/gate_report.json"
OUT_KERNEL = ROOT / "ai/kernels/bovine-lsd-gate/out/gate_report.json"


def main() -> None:
    lines = PROGRESS.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines)

    def grab(pattern: str, flags=0):
        m = re.search(pattern, text, flags)
        if not m:
            raise SystemExit(f"not found in progress.txt: {pattern}")
        return m

    m = grab(r"^5: features \((\d+), 960\) in (\d+)s", re.M)
    n_items, feat_secs = int(m.group(1)), int(m.group(2))

    m = grab(r"^5: SHIPPED lesion model on non-animals -- in-domain held-out n=(\d+): flags ([\d.]+)%;\s+novel n=(\d+): flags ([\d.]+)%", re.M)
    before_gate = {
        "in_domain_heldout": {"n": int(m.group(1)), "flagged_rate": float(m.group(2)) / 100},
        "novel": {"n": int(m.group(3)), "flagged_rate": float(m.group(4)) / 100},
    }

    m = grab(r"^6: head trained, epochs (\d+), val acc ([\d.]+)", re.M)
    head_train = {"epochs": int(m.group(1)), "val_acc": float(m.group(2))}

    m = grab(r"^7: OOD fit on (\d+) bovine train features", re.M)
    n_fit = int(m.group(1))

    k_rows = re.findall(
        r"^7: k=\s*(\d+)\s+AUROC\(other-test vs bovine-test\)=([\d.]+)\s+median d2 bovine-test ([\d.]+) other-test ([\d.]+) novel ([\d.]+)",
        text, re.M)
    k_candidates = {k: {"auroc_other_test_vs_bovine_test": float(a), "median_d2": {"bovine_test": float(b), "other_test": float(o), "novel": float(n)}}
                    for k, a, b, o, n in k_rows}
    K = int(grab(r"^7: chosen k=(\d+)", re.M).group(1))

    m = grab(r"^8: head held-out acc (\{.*?\})\s+confusion (\[\[.*?\]\])", re.M)
    heldout_acc, confusion = json.loads(m.group(1)), json.loads(m.group(2))
    species = json.loads(grab(r"^8: species (\{.*\})$", re.M).group(1))

    m = grab(r"^9: combined model built; d2 in-graph vs numpy rel err ([\d.e+-]+)", re.M)
    d2_rel = float(m.group(1))

    m = grab(r"^10: exported\s+outputs (\[.*?\])\s+fp32 ([\d.]+) MB\s+fp16 ([\d.]+) MB", re.M)
    outputs = json.loads(m.group(1).replace("'", '"'))
    sizes_mb = {"fp32": float(m.group(2)), "fp16": float(m.group(3))}

    verification = json.loads(grab(r"^11: (\{\"lesion\".*\})$", re.M).group(1))
    verification["novel_rejected_fp16_combined"] = json.loads(
        grab(r"^11: novel \(fp16, combined\) (\{.*\})$", re.M).group(1))

    report = {
        "modelVersion": "bi-lsd-mnv3l-v1.0.0-gate",
        "reconstructed_from": "ai/kernels/bovine-lsd-gate/out/progress.txt (kernel v5); the kernel's own "
                              "json.dump failed on a name collision after every computation had finished",
        "absent_stdout_only": ["ood.sweep (D per keep level -- recomputed locally by verify_gate.py)",
                               "head.softmax_only_sweep", "combined_sweep.novel_by_source"],
        "outputs": outputs,
        "data": {"n_items_featurised": n_items, "feature_extraction_seconds": feat_secs,
                 "ood_fit_n_bovine_train": n_fit,
                 "in_domain_other_heldout_n": before_gate["in_domain_heldout"]["n"],
                 "novel_n": before_gate["novel"]["n"],
                 "novel_sources": ["intel scenes (buildings, forest, glacier, mountain, sea, street)",
                                   "LFW faces", "Caltech-101 classes never trained on (every third class)"]},
        "lesion_model_on_nonanimals_before_gate": before_gate,
        "head": {"training": head_train, "heldout_accuracy": heldout_acc, "confusion": confusion, "species": species},
        "ood": {"k": K, "k_candidates": k_candidates, "in_graph_vs_numpy_rel_err": d2_rel},
        "verification": verification,
        "verification_complete": True,
        "sizes_mb": sizes_mb,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_KERNEL.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  k={K}  AUROC={k_candidates[str(K)]['auroc_other_test_vs_bovine_test']}")
    print(f"  species: cattle {species['cattle_as_cattle']:.4f}  buffalo {species['buffalo_as_buffalo']:.4f}")
    print(f"  novel rejected (fp16, combined): {verification['novel_rejected_fp16_combined']}")
    print(f"  shipped model flagged novel non-animals as diseased: {before_gate['novel']['flagged_rate']:.1%}")


if __name__ == "__main__":
    main()
