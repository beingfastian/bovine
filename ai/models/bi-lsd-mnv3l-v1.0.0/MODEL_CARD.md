# Model Card — `bi-lsd-mnv3l-v1.0.0`

**BovineInsight AI-assisted livestock health screening**
Binary lumpy-skin-lesion screening for cattle photographs.

**Trained:** 2026-09-20 · **Status:** approved for **collect-only / screening** use in V0
**Replaces:** `finals_models00.tflite` (failed Phase A — see [Verdict on the predecessor](#verdict-on-the-predecessor))

---

## 1. What it is

| | |
|---|---|
| Architecture | MobileNetV3-Large (ImageNet), 1-unit sigmoid head |
| Parameters | 3,119,489 |
| Artifact | `bi-lsd-mnv3l-v1.0.0.tflite`, **6.24 MB**, float16 |
| **Input** | **raw `[0,255]` RGB, 224×224×3, NHWC** |
| **Output** | **`[1,1]` float32 — already a sigmoid probability** |
| Operating threshold | **τ = 0.50** |
| Calibration | **none applied** (see §5) |

### Input/output contract — read this before writing any client code

**Feed raw `[0,255]` pixel values.** The graph's first two ops are the Keras
`Rescaling` layer (`MUL 0.00784314` = 1/127.5, then `ADD -1.0`), mapping
`[0,255] → [-1,1]` internally. **Do not** divide by 255. **Do not** apply ImageNet
mean/std. That mismatch is precisely what destroyed the predecessor.

**The output is already a probability.** The final graph op is `LOGISTIC`.
`p = output[0][0]` is P(lesion); P(normal) is `1 - p`. Apply **no** softmax and
**no** sigmoid — a second activation would squash `[0,1]` into `[0.5,0.73]` and
silently invalidate τ.

---

## 2. Intended use

**In scope**
- Preliminary screening aid inside the three-gate wrapper (animal presence →
  image quality → confidence/OOD → three-way verdict)
- Flagging animals for veterinary attention
- Generating screening records that accumulate the field dataset

**Out of scope**
- Diagnosis. This is not a diagnostic device.
- Naming a disease to the user. V0 says *"possible skin condition"*, never
  *"Lumpy Skin Disease"*.
- Rendering the word **"Healthy."** The negative verdict is about the *photograph*
  — *"no obvious skin lesion found in this photo"*.
- Any treatment or medication recommendation.
- **Buffalo.** See §6.

---

## 3. Training data

**2,798 groups** from 3,030 images. Report the **group** count, never the file count.

| Source | Licence | Contribution |
|---|---|---|
| [Mendeley `w36hpf86j2`](https://data.mendeley.com/datasets/w36hpf86j2/1) — Kumar & Shastri, 2022 | CC BY 4.0 | 1,024 images → 907 groups |
| [Roboflow `cattle_disease-detection`](https://universe.roboflow.com/mdzillur-rahaman-rohan/cattle_disease-detection) v11 | CC BY 4.0 | 7,014 files → 2,013 source images |
| [Zenodo `LumpySkinDisease_DataHub`](https://zenodo.org/records/11003967) | CC BY 4.0 | **severity labels only** — byte-identical images to Mendeley |

| | Groups |
|---|---|
| Lesion | 1,833 |
| Normal | 965 |
| Imbalance | 0.53 : 1 |

### Three layers of deduplication

1. **Roboflow augmentation stripping.** The v11 export is 8,014 files from only
   ~2,784 source images (train is 3.6× augmented). Removed deterministically by
   filename stem — pHash does *not* reliably collapse flips and rotations, so
   augmented siblings would otherwise have leaked across folds.
2. **Exact duplicates** (SHA-256): 5 removed.
3. **Near-duplicates** (pHash, Hamming ≤ 5): 162 clusters, 394 images,
   **75 of them cross-source**.

### Data defects found and corrected

- **3 images byte-identical across both `Lumpy Skin/` and `Normal Skin/`** in the
  published Mendeley set. Visual inspection confirmed clear nodular lesions in
  all three; the `Normal` copies are mislabelled. Resolved to label 1.
- **2 near-duplicate groups with conflicting labels.** One relabelled to lesion
  (unambiguous nodules), one dropped as ambiguous. Decisions recorded as data in
  `ai/data/label_overrides.csv`, not as manual edits.

### Splits

Group-aware and stratified on the **joint (label, source)**, not label alone,
because source nearly determines label:
`P(lesion | mendeley) = 0.299` vs `P(lesion | roboflow) = 0.826`.

| Split | Groups | P(lesion) |
|---|---|---|
| test (locked) | 559 | 0.657 |
| fold0–4 | ~448 each | 0.653–0.656 |

Label proportion spread across all six splits: **0.003**.

### Held out entirely

Roboflow's `Infected_Foot_Image`, `Mouth Disease Infected` and
`Normal_Mouth_Image` classes (1,000 images) were **excluded from training** and
reserved as an out-of-distribution suite — other diseases the model must route to
"unclear" rather than confidently call lumpy skin. **Not yet evaluated.**

---

## 4. Performance

### Cross-validation (5-fold, group-aware)

| Metric | Mean ± SD | 95% CI |
|---|---|---|
| AUC | 0.9640 ± 0.0050 | ±0.0044 |
| Balanced accuracy | **0.9148 ± 0.0078** | ±0.0069 |
| Recall | 0.9400 ± 0.0096 | ±0.0084 |
| Specificity | 0.8896 ± 0.0219 | ±0.0192 |

### Locked test set — evaluated once (n = 611)

| Metric | Value |
|---|---|
| AUC | **0.9653** |
| Balanced accuracy | **0.9090** |
| Recall | 0.9629 |
| Specificity | 0.8551 |
| Precision | 0.9284 |
| TP / TN / FP / FN | 389 / 177 / 30 / **15** |

**Test balanced accuracy (0.9090) sits within one CV standard deviation of the
cross-validated mean (0.9148 ± 0.0078).** Held-out performance matching
cross-validation is the strongest available evidence that no leakage survived the
deduplication and splitting pipeline.

### Against baselines

| Baseline | Balanced accuracy |
|---|---|
| Previous model (`finals_models00.tflite`) | 0.6253 |
| Majority class ("always lesion") | 0.6612 |
| **Source shortcut** (predict dataset, not disease) | **0.7556** |
| **This model** | **0.9090** |

### Per source (test)

| Source | n | AUC | Balanced acc | Recall |
|---|---|---|---|---|
| mendeley | 205 | 0.9558 | 0.8969 | 0.8571 |
| roboflow | 406 | 0.9368 | 0.8297 | 0.9824 |

**Why this matters.** A classifier trained to predict *source* rather than disease
reaches **AUC 0.9596** — the datasets are nearly separable. Because source almost
determines label, dataset recognition alone yields 0.7556 balanced accuracy. The
per-source AUCs above (0.9558 / 0.9368) are measured *within* a single source,
where that shortcut carries zero information. **The model is not taking it.**

### Per severity (test)

| Severity | n | Recall | 95% CI |
|---|---|---|---|
| Mild | 47 | **0.8511** | ±0.1018 |
| Severe | 15 | 0.9333 | ±0.1262 |

`n=15` for Severe is too small to quote as a figure. Mild spans [0.75, 0.95].

Severity labels come from the Zenodo re-annotation of the Mendeley images, so
**these figures describe Mendeley-source photographs only.**

---

## 5. Threshold and calibration

**τ = 0.50**, selected on the **Mild-recall curve**, not on balanced accuracy.

| τ | Mild recall | Severe recall | Specificity |
|---|---|---|---|
| 0.74 (bal-acc optimum) | 0.7542 | 0.9867 | 0.8860 |
| 0.55 | 0.8212 | 0.9867 | 0.8346 |
| **0.50 (shipped)** | **0.8659** | **1.0000** | **0.8162** |
| 0.45 | 0.8715 | 1.0000 | 0.8015 |

0.50 is the knee: +4.5 points of early-case recall for 1.8 points of specificity.
Below it, returns collapse.

**Rationale.** A false negative tells a farmer a diseased animal is fine — with a
WOAH-listed notifiable disease. A false positive costs one veterinary
consultation, which is *already* the app's recommended action. The asymmetry
justifies the trade.

**Calibration: measured, not applied.** Uncalibrated ECE is **0.0302**, five times
inside the 0.10 bar. Temperature scaling (T = 1.2097) would reach 0.0240 — a
negligible gain in exchange for an additional client-side transform, in a codebase
that has already shipped one broken preprocessing pipeline. **Ship raw sigmoid
output.**

---

## 6. Limitations

### Buffalo — the largest gap

- **59.4% of Pakistan's milk is buffalo milk** (44.4 of 74.7 bn litres);
  47.74M buffaloes nationally, 31.89M in Punjab.
- **Every training image is cattle. No buffalo data exists, public or otherwise.**
- LSD presents *differently* in buffalo — small nodular lesions **without** the
  centred ulcerations seen in cattle.
- Nili-Ravi and Kundi buffalo are black-hided; nodule detection depends on
  contrast and shadow. (Gamma augmentation 0.5–1.8 was applied as partial
  mitigation. It is not a substitute for data.)

### Buffalo — specificity MEASURED 2026-09-20, sensitivity still unknown

Tested against [Buffalo-Pak](https://data.mendeley.com/datasets/vdgnxsm692/2)
(Mendeley `vdgnxsm692`, CC BY 4.0) — **324 healthy Pakistani buffalo**, all
positives therefore false alarms:

| Breed | n | False-positive rate | median p |
|---|---|---|---|
| Khundi | 75 | 0.0800 | 0.1103 |
| Mixed | 137 | 0.1022 | 0.0856 |
| **Nili-Ravi** | 112 | **0.1786** | 0.1246 |
| **All buffalo** | **324** | **0.1235** | **0.1046** |
| *cattle test set (reference)* | *611* | *0.1449* | — |

**Specificity on Pakistani buffalo is 0.8765 — slightly better than the 0.8551
measured on the cattle test set.** The model does not mistake black buffalo hide
for pathology. Maximum score across all 324 was 0.8767, whereas diseased cattle
routinely exceed 0.99; the buffalo distribution sits low throughout.

Nili-Ravi is the weakest breed at roughly 2x Khundi's false-alarm rate, so the
dark-hide concern is real but modest.

Visual review of the 18 highest-scoring buffalo found **no genuinely affected
animals** — all 40 positives are true false alarms. Two likely contributors are
visible: several are rotated 90 degrees (EXIF orientation in the source data,
out-of-distribution for a model trained on upright animals), and many are shot
against brick and cobblestone, whose texture plausibly reads as nodular.

> **SENSITIVITY ON BUFFALO REMAINS ENTIRELY UNKNOWN.** Buffalo-Pak contains no
> diseased animals, so this shows only that the model does not cry wolf — not
> that it would detect a sick buffalo. **A model that flagged no buffalo at all
> would score the same 0.8765.** The false-negative direction is the dangerous
> one, and it is unmeasured. Buffalo are still **out of scope** for any
> detection claim.

### Early-stage detection is materially weaker

Grad-CAM on 30 correctly-classified lesion images shows **broad, whole-body
activation rather than focal attention on nodules**. The model appears to read
*"this animal looks generally affected"* rather than *"these are discrete
nodules"*.

This predicts, and is confirmed by, the severity gap: Severe recall 0.99 vs Mild
recall 0.75 (at τ=0.74). Two independent measurements, one explanation.

### Training data quality

Inspection of Grad-CAM tiles surfaced, in the training set:
- **Images where the model attends to people, not animals** (≥2 of 30)
- **News screenshots with overlay text** (e.g. "…AL FAIRS / ED DUE TO…")
- **Photo collages** — mosaics of 4+ separate images stitched together

All web-sourced, with no documented collection methodology in any source dataset.

### No field validation

Every image is web-sourced. **Nothing here predicts performance on a phone camera
in a Punjab shed** — dust, harsh sun, motion blur, real working distances.

**Acceptance test T3 — ≥250 vet-labelled photographs from real Pakistani farms,
covering cattle *and* buffalo — remains unstarted and remains the only measurement
that determines production readiness.**

---

## 7. Verdict on the predecessor

`finals_models00.tflite` (the FYP model) was measured in Phase A on all 1,024
Mendeley images:

- Fed its own verified contract (raw `[0,255]`): **a constant `0.0000` for every
  image**, ROC-AUC exactly 0.5000, all 324 diseased animals called normal.
- Best over six preprocessing conventions: **0.6253 balanced accuracy** — below
  the 0.6836 majority-class baseline. **A constant function outperformed it.**

**Root cause:** trained on `[0,1]`-normalised images fed into a network already
containing `include_preprocessing=True`, compressing all training data into
`[-1, -0.992]` — a 0.8%-wide slice of the input range. The bug was encoded in the
weights and unreachable by any inference-time fix.

The architecture choice was sound and was retained. The weights were discarded.

---

## 8. Reproducibility

```
ai/scripts/dedupe.py          3-layer deduplication -> manifest.csv
ai/scripts/make_splits.py     group-aware, (label,source)-stratified splits
ai/scripts/build_bundle.py    256px training bundle (1 GB -> 87 MB)
ai/notebooks/bovine_lsd_training.ipynb   training + evaluation
ai/reports/phase-a.md         predecessor failure analysis
ai/reports/dedupe.md          deduplication numbers
ai/reports/splits.md          split composition
```

Seed 42 throughout. Source archives SHA-256 verified.

### Guards built into the pipeline

- `assert_input_range()` runs before **every** `fit()` — fails loudly if pixels
  are not in `[0,255]`. One line; it makes the predecessor's failure impossible.
- `base(x, training=False)` keeps BatchNorm in inference mode during fine-tuning.
- Hard `assert`s on group leakage between test and every fold.

### Export verification

| Check | Result |
|---|---|
| Size | 6.24 MB (bar ≤ 8 MB) |
| Input / output shape | `[1,224,224,3]` / `[1,1]` float32 |
| Keras↔TFLite max abs diff | 0.0077 |
| **Decision agreement at τ=0.50** | **1.0000** (0/100 samples near threshold) |

The 0.0077 difference is expected float16 accumulation and has **zero** effect on
any decision at the shipped threshold.

### ONNX export for the browser — verified 2026-09-21

`bi-lsd-mnv3l-v1.0.0.fp16.onnx` (opset 13, float16 weights, **fp32 input and
output**) runs client-side under `onnxruntime-web`. The input/output contract in
§1 is unchanged.

| Check (locked test set, n=611) | Result |
|---|---|
| ONNX (fp32) ↔ Keras max abs diff | **1.8e-06** |
| ONNX (fp32) ↔ TFLite decision agreement at τ | **1.0000** |
| Confusion matrix vs §4 | **identical** (389/177/30/15) |
| Balanced accuracy / AUC | **0.9090 / 0.9658** |
| fp16 ↔ fp32 decision agreement | **1.0000** (bal. acc 0.9090, max diff 0.0088) |
| Size | 6.32 MB (5.73 MB gzipped) |

**The first export attempt was broken and the check caught it.** `model.export()`
traced the training-time `augment` block, leaving five `StatelessRandomUniformV2`
nodes in the graph — invalid ONNX, and had it loaded it would have randomly
flipped and recoloured every photo at inference. The shipped artifact is exported
from an inference-only rebuild (backbone + head, augmentation and dropout
removed), proven equivalent to `model.predict()` before export. The converter now
asserts that no random or control-flow ops survive.

**int8 quantisation was built and rejected**: 2.92 MB gzipped, but decision
agreement fell to **0.8101** and balanced accuracy to **0.7048**. MobileNetV3's
depthwise convolutions and hard-swish activations do not survive 8-bit
quantisation. Numbers in `ai/reports/quantization.json`.

Browser-side preprocessing is a port of Pillow's bilinear filter
(`web/lib/resize.ts`), not canvas `drawImage()`. Measured on real fixtures,
canvas resampling drifted up to **0.027** from the Python reference and flipped a
verdict; the port reduced that to **0.0073**. `/selftest` re-measures this on any
device.

```
ai/scripts/build_onnx_notebook.py   generates the Kaggle conversion kernel
ai/scripts/verify_onnx.py           local ONNX↔TFLite parity + threshold sweeps
ai/scripts/quantize_onnx.py         fp16/int8 candidates, measured not assumed
ai/scripts/make_selftest_fixtures.py  browser parity fixtures
ai/reports/onnx_verification.json   the numbers above
```

### Gate 1 and species detection — `bi-lsd-mnv3l-v1.0.0-gate` (2026-09-23)

Field testing on 2026-09-22 produced a lesion verdict for a photograph of a
**hand**. The shipped model, measured afterwards, flagged **12–24%** of
non-animal photographs as "possible condition". Gate 1 from
`AI_IMPLEMENTATION_PLAN.md` §6.2 was therefore built, as two extra outputs on
the same backbone (download +1.0 MB; lesion weights byte-identical):

| Output | What | Held-out result |
|---|---|---|
| `species` | Dense(3) softmax: cattle / buffalo / other | cattle→cattle **0.9967**, buffalo→buffalo **1.0000** (all three breeds), other→other 0.9885 |
| `ood` | Mahalanobis d² of the 960-d pooled feature from the bovine training distribution, PCA k=384, computed in-graph | AUROC non-animal vs animal **0.956** |

**Why two halves.** The first head, trained against only eight negative
categories, was tested on twenty real landscape photographs and rejected
**none** — every one scored p(cattle) = 1.000. The fine-tuned backbone learned
that fields, grass and sky mean cattle, and a closed softmax cannot say "none
of these" (§6.1 of the plan predicted exactly this). Broadening the negatives
to ~85 categories fixed most of that; the distance term covers what remains.
With both, **12/12** of those landscapes are rejected.

**Honest negatives, and an honest decomposition.** Training negatives:
natural-images, real hand photographs, 70% of Caltech-101's classes.
Everything the gate is judged on for open-set behaviour was never trained on in
any form — Intel scenes, LFW faces, the other 30% of Caltech classes
(class-disjoint). On **6,588** such photographs, at the shipped operating
point:

| Half | Novel non-animals rejected |
|---|---|
| distance (`ood ≥ 746`) alone | **29%** (Caltech-novel 40%, scenes 28%, faces 8%) |
| softmax (`p(other) ≥ 0.8`) alone | **~75%** (80% at 0.7, 67% at 0.9) |
| **combined (shipped)** | **78.4%** |

The softmax carries most of the load once its negatives are broad enough. The
distance term adds three to four points overall, and it is the half that
catches high-resolution outdoor scenes — 2 of the 12 local landscapes were
rejected by it alone. Neither is a detector; the remaining ~22% of novel
non-animals will still be scored, which is why the rejection screen asks the
person whether an animal is really there and why every row stores both
scores raw.

**Operating point.** `oodMax = 746.1` (99.5th percentile of held-out cattle
d²), `otherMax = 0.8`. Joint real-animal retention on the locked test split:
**cattle 0.9902, buffalo 1.0000**. The six cattle photographs turned away were
inspected individually (`ai/reports/gate_rejected_cattle_inspection.json`): a
laptop screenshot with overlay text (labelled *lesion*), a cartoon cow, a
fibreglass statue, an antelope, a distant herd, and one genuine loss — calves
in coats at d² = 747. Four of the six are the training-data contamination §6
already documented; the gate now catches it at inference.

**Not measured:** field photographs. Every number above is web-sourced or
curated. The gate stores `p_other` and `ood_distance` raw on every field row
precisely so both thresholds can be retuned from real data.

```
ai/scripts/build_gate_notebook.py     generates the Kaggle kernel (train + export + verify)
ai/scripts/verify_gate.py             local: lesion unchanged, thresholds, real non-animals
ai/reports/gate_report.json           held-out numbers (reconstructed from the kernel's progress log)
ai/reports/gate_verification.json     the local checks
```

---

## 9. Acceptance status

| Test | Bar | Result |
|---|---|---|
| T2 — CV balanced accuracy | ≥ 0.85 | **0.9148** ✅ |
| T7 — ECE | ≤ 0.10 | **0.0302** ✅ |
| T9 — per-source metrics | reported separately | ✅ |
| T11 — model size | ≤ 8 MB | **6.24 MB** ✅ |
| Export parity | decisions agree | **1.0000** ✅ |
| T8 — Grad-CAM audit | ≥ 25/30 focal on lesions | **marginal** ⚠ |
| T4 — buffalo specificity | ≥ 100 images | **324 tested, FPR 0.1235** ✅ |
| T4b — buffalo **sensitivity** | ≥ 100 diseased | **0** ❌ |
| T5 — abstention rate | ≤ 35% | not measured |
| T6 — OOD suite | ≥ 90% routed to unclear | not run |
| T10 — on-device latency | p95 ≤ 1.5 s | not measured |
| T13 — vs public baseline | ≥ baseline | not run |
| **T3 — field test set** | **≥ 250 vet-labelled** | **not started** ❌ |

**Approved for collect-only / screening use in V0, inside the three-gate wrapper,
with the §6 limitations stated in-product.**

**Not approved for any diagnostic claim, disease naming, or buffalo
DETECTION.** Buffalo specificity is validated; buffalo sensitivity is not.

---

## 10. Next version

**v1.1 — lesion-crop training.** The Roboflow export ships YOLO bounding boxes
that have not been used. Training on lesion crops rather than whole animals would
force focal learning and directly attack the Mild-recall gap identified in §6.
The data is already on disk. **Highest-value experiment available.**

Also queued: purge collages and news screenshots; run the T6 OOD suite on the
held-out foot/mouth classes; measure on-device latency; benchmark against
`qq/lumpy-skin-disease-detection`.

**And above all of it: T3.** No amount of public-data engineering substitutes for
250 vet-labelled photographs from Pakistani farms, cattle and buffalo both.
