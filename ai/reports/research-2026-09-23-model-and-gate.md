# Research: where the model actually stands, and what would move it

**Date:** 2026-09-23 · **Status:** for review before any weights change
**Trigger:** field testing scored a photograph of a hand; the first gate head passed
20/20 landscapes as cattle; the request was to stop iterating and understand first.

This document reads the literature against our own numbers. Every claim about our
model cites a report file in this repo; every claim about the field cites a paper
listed in §10. Two sources could not be read (publisher walls) and are marked.

---

## 1. Summary in six sentences

1. **Our lesion model is not behind the literature; the literature is ahead of
   itself.** Every LSD paper found trains and reports on the same two web datasets
   we use, with no held-out test set, no de-duplication, and no field images; the
   99% results come from splitting the Roboflow export *after* its 3.6× augmentation.
   Our 0.909 balanced accuracy on a group-deduplicated, locked, once-evaluated test
   split is the more credible number, and chasing theirs would mean reintroducing
   the leakage we removed.
2. **The one thing that would genuinely move the real number is field data** — the
   model card's T3 (≥250 vet-labelled farm photographs, cattle and buffalo). Nothing
   read here changes that; several sources sharpen it.
3. **Our training recipe already matches best practice** (linear-probe-then-fine-tune,
   BatchNorm frozen, 100× learning-rate drop, class weights, early stopping on AUC).
   The gaps are second-order: Balanced MixUp is worth one controlled trial; the
   unfreezing depth (40 layers) was never swept.
4. **The known weakness — mild-case recall, whole-body rather than focal attention —
   has a literature answer:** patch-based / multiple-instance training on lesion crops
   raised mean sensitivity by 7% on human skin lesions and improved robustness to
   distribution shift. We already hold the bounding boxes (Roboflow) to do this.
   It is the highest-value model experiment and was already queued as v1.1.
5. **On the gate, the literature explains both what happened and why the fix worked
   the way it did:** fine-tuning distorts pretrained features and hurts
   out-of-distribution behaviour (Kumar et al.), so distance scores on our backbone
   are inherently weak (29% alone); training against diverse outliers — "outlier
   exposure", which is exactly what the broadened `other` class is — is the strongest
   lever (~75% alone). One cheap, well-evidenced improvement remains untested:
   ℓ2-normalising features before the Mahalanobis distance (Mahalanobis++).
6. **Two things we did not add were right not to add:** test-time augmentation
   (degrades medical image classifiers in 11 of 12 tested cases) and post-training
   int8 quantisation (fails on MobileNetV3's hard-swish; needs QAT).

---

## 2. What the LSD literature reports, and what it did

| Paper | Data | Split | Dedup | Field / buffalo | Best claim |
|---|---|---|---|---|---|
| Comparative pretrained models (PMC11512320) | Mendeley 1,024 + Roboflow 2,223 | **90/10 train/val, no test set** | none stated | none | VGG16 96.07%; MobileNetV2 96.39% |
| MobileNetV2 + RMSprop (PLOS One 2024) | GitHub, **793 images** | 563 / 230 | none stated | none | 95% acc, **5 epochs** |
| ConvNeXtLSD multi-class (PMC13507278) | Roboflow export, **8,014 images** | 88/6/6 stratified | **none** — and the 8,014 are 3.6× augmented copies of ~2,784 sources | none | 98.29% acc, 87.6M params |
| ViT mobile app (Sci. Rep. 2025) | online + field, counts unread | unread | unread | field images claimed | 98.12% (not verified — paywalled) |

Three observations that matter for us:

- **The 8,014-image dataset is the Roboflow v11 export.** Our deduplication report
  found it contains ~2,784 source images with train augmented 3.6×; splitting it
  after augmentation puts rotated/flipped copies of test images in training. A 98%
  accuracy on that split is a measurement of augmentation leakage. Our pipeline
  strips augmentation by filename stem *before* splitting
  (`ai/reports/dedupe.md`).
- **The error analysis in the MobileNetV2 paper is our error analysis.** Their model
  failed on "images taken from considerable distance", "surrounding context and
  complexity", and "black colour being incorrectly interpreted as a lump". Those
  are, respectively, our distant-herd rejection, our brick/cobblestone false alarms
  on Buffalo-Pak, and the Nili-Ravi dark-hide false-alarm rate — with the
  difference that we measured them (`MODEL_CARD.md` §6).
- **Nobody has buffalo.** No paper found mentions buffalo images at all. Our 324
  Buffalo-Pak images are, as far as this search can tell, the only buffalo
  evaluation in the published record, and they contain no diseased animals. The gap
  the project exists to close is real and unaddressed anywhere.

The methodological-failures literature (Varoquaux & Cheplygina, npj Digital Medicine
2022 — abstract read via EuropePMC, full text paywalled) names exactly these
patterns: leakage through duplicates, absent or tiny test sets, and evaluation on
curated rather than clinical data. Our runbook already cites it.

**Implication.** Do not retarget the lesion model at a higher headline number.
Retarget it at the two things the card says are weak — mild-case recall and field
generalisation — and measure them the way we already do.

---

## 3. Our training recipe against best practice

What `ai/scripts/build_notebook.py` actually did:

| Element | Ours | Keras transfer-learning guide / literature |
|---|---|---|
| Base | MobileNetV3-Large, ImageNet, `include_preprocessing=True` | ✓ |
| BatchNorm | `base(x, training=False)` throughout | ✓ exactly the guide's warning |
| Stage 1 | head only, Adam 1e-3, ≤15 epochs | ✓ |
| Stage 2 | unfreeze top 40 layers, Adam 1e-5 (100× lower), ≤25 epochs | ✓ — this is **LP-FT** (Kumar et al. 2022): linear probe first, then fine-tune, which they show beats plain fine-tuning both in- and out-of-distribution |
| Imbalance | balanced class weights | reasonable; benchmark (PMC13511939) found no universal winner, **Balanced MixUp** best for EfficientNet-class nets, undersampling and GAN augmentation harmful |
| Early stopping | on `val_auc`, patience 8, restore best | ✓ (not accuracy — the runbook's own point) |
| LR schedule | ReduceLROnPlateau | adequate; warmup+cosine is a transformer convention, not evidenced to matter here |
| Augmentation | flip, rotation 0.06, zoom 0.2, brightness 0.3, contrast 0.3, gamma 0.5–1.8 | ✓; gamma for dark hides is a defensible, documented mitigation |
| Resolution | 224 | papers use 224–256; no evidence either way for us |
| Label smoothing / mixup | none | small, real regularisation effect; mixup also improves calibration (Thulasidasan 2019) |

**Verdict:** the recipe is sound and matches the guidance we could find. There is no
"obvious mistake" to fix in how the lesion model was trained. The two second-order
experiments worth one controlled run each: Balanced MixUp, and an unfreezing-depth
sweep (20 / 40 / 80 layers). Neither is expected to move balanced accuracy by more
than a point or two, and neither addresses the real weaknesses.

---

## 4. The real weakness, and its literature answer

`MODEL_CARD.md` §6: Grad-CAM shows whole-body activation, not focal attention on
nodules; Mild recall 0.85 vs Severe 0.93; the model reads "this animal looks
affected", not "these are nodules".

The human skin-lesion literature met this exact problem:

- **Patch-based attention** (Gessert et al., arXiv 1905.02793): high-resolution
  patches with learned global context "improves mean sensitivity by 7%" over
  whole-image baselines; cropping beat resizing; class balancing "significantly
  improves the mean sensitivity".
- **Key patches / multiple-instance learning** (Araújo et al., CVPR-W 2024): forcing
  the model to use a small subset of patches did not cost in-domain performance and
  was "more robust to shifts in patient demographics" — i.e. exactly the
  curated-web → farm-photo shift we face — with region-level explanations for free.
- **ConvNeXtLSD** itself lists as its main failure mode "subtle symptom manifestation
  … localized lesions", and proposes attention mechanisms as future work.

We already have what this needs: the Roboflow export ships YOLO bounding boxes for
lesions that our pipeline has never used (`MODEL_CARD.md` §10, "v1.1 — lesion-crop
training … highest-value experiment available"). The research confirms that
judgement rather than changing it.

**Design for v1.1 (for review, not started):** train the same backbone on lesion
crops plus matched negative crops; at inference, score a grid of crops of the animal
region and aggregate (max, or top-k mean) — MIL-style — so a single focal nodule can
drive a positive. Evaluate once on the locked test split with per-severity recall;
the number that has to move is **Mild recall (0.85)**, and specificity must not fall
below the current 0.855.

---

## 5. Gate 1 / open-set detection: what the literature says about what we saw

**What we saw.** An 8-category softmax head passed 20/20 landscapes as cattle
(p=1.000). Broadening negatives to ~85 categories made the softmax reject ~75% of
6,588 never-trained-on non-animals alone; a Mahalanobis distance on the same features
rejected 29% alone; combined 78.4%, at 99.0% cattle / 100% buffalo retention
(`ai/reports/gate_report.json`).

**What the literature says, in order of how much it explains:**

1. **Fine-tuning distorts pretrained features** (Kumar et al., ICLR 2022): fine-tuned
   models gain ~2% in-distribution and lose ~7% out-of-distribution versus a linear
   probe, because fine-tuning updates the in-distribution subspace
   disproportionately. Our backbone was fine-tuned for lesions on cattle; its 960-d
   features are optimised to separate lesion from normal *within* cattle photos,
   not to place landscapes far from cattle. **A distance score on these features is
   structurally handicapped.** 29% is what that looks like.
2. **Outlier exposure is the strongest single lever** (Hendrycks et al., ICLR 2019):
   training with an auxiliary set of diverse outliers "significantly improves"
   detection of *unseen* anomalies. Our `other` class is outlier exposure by another
   name, and its jump from 0/20 to ~75% when the outlier set went from 8 to ~85
   categories is the paper's central finding reproduced. **Diversity of the outlier
   set is the parameter that mattered**, not the scoring function.
3. **There is no universally best post-hoc score, and rankings are fragile**
   (OpenOOD v1.5; Szyc et al. 2023): method ranking changes with backbone,
   training details, random seed and OOD dataset choice. The only evaluation that
   means anything is on the deployment's own novel set — which is why
   `gate_report.json` and the local wallpaper check exist, and why the Intel-scenes
   result (28% at 150 px) and the local high-res landscapes (12/12) can both be
   true.
4. **Plain Mahalanobis is weakened by feature-norm variation** (Mahalanobis++,
   Mueller & Hein 2025): "strong variations in feature norms" violate the Gaussian
   assumption; ℓ2-normalising the features first "improves … significantly and
   consistently" across 44 models, post-hoc, no training. **We do not normalise.**
   This is one extra layer in the graph and one kernel run to measure.

**Implication for the gate, ranked by expected value per hour:**

| # | Change | Why | Cost | Expected |
|---|---|---|---|---|
| G1 | ℓ2-normalise features before PCA/Mahalanobis | Mahalanobis++ | 1 layer, 1 kernel run | ood-only 29% → materially higher; measured on the same novel set |
| G2 | Add *targeted* outliers: brick/cobble walls, sheds, farm backgrounds without animals, hands, feet, mud | outlier exposure; model card's own false-alarm review | dataset curation | softmax-only ↑ on the categories a farmer will actually photograph |
| G3 | Evaluate KNN-on-normalised-features and MaxLogit/energy from the 3-way head as alternative/complementary scores | OpenOOD: try several, judge locally | kernel cells | pick by novel-set AUROC, not by name |
| G4 | Ask "is there actually an animal?" on rejection (shipped) | turns false rejections into labelled data | done | — |

Not recommended: a second general-purpose ImageNet classifier as the gate (+5–10 MB;
judges close-up skin photos as "not an ox"); a training-time OOD method that
retrains the backbone (risks the lesion output, which must stay byte-identical).

---

## 6. Thresholds, prevalence and what the score means in a field

- **Youden's J is not neutral** (Schisterman & Perkins / BMC 2010): maximising J
  implicitly sets the false-negative : false-positive cost ratio to (1 − prevalence),
  so it silently changes with the population. Our τ = 0.50 was chosen on the
  Mild-recall knee with an explicit cost argument (a false negative is a notifiable
  disease; a false positive is one vet visit) — the decision-theoretic route the
  paper recommends, not J.
- **Prevalence will fall off a cliff in the field.** The test split is 65.7%
  lesion. A screening population might be 2–20%. Sensitivity and specificity do not
  move with prevalence; **positive predictive value does**. At 0.963 recall / 0.855
  specificity: PPV ≈ **12%** at 2% prevalence, **42%** at 10%, **74%** at 30%. Most
  positives in a low-prevalence herd will be false alarms. This is inherent to any
  screening test at these operating characteristics; it must be stated in-product
  ("most flagged animals will turn out fine — the point is not to miss one"), and it
  is why the verdict copy never says more than "possible skin condition".
- **Abstention is the right structure** (Chow's rule; selective classification): a
  dead-band around τ plus gates that abstain. Ours is measured
  (`ai/reports/onnx_verification.json`). Field data should retune the band from the
  observed abstention rate (card T5: ≤35%).

---

## 7. Things the literature says *not* to do (and we did not)

- **Test-time augmentation by default.** Across MedMNIST tasks TTA degraded accuracy
  in 11 of 12 model–dataset pairs, up to 31.6 points, worst for BatchNorm CNNs with
  geometric transforms (arXiv 2604.09697). If ever tried: intensity-only, include the
  unaugmented view, validate on the locked test split first.
- **Post-training int8 quantisation of MobileNetV3.** PTQ "fails to quantize
  MobileNetV3 properly due to hard-swish"; QAT recovers it. Our measured int8 collapse
  (agreement 0.81) is this. fp16 was the right call; if 7 MB ever matters, QAT is the
  route, not PTQ.
- **Chasing the 98–99% numbers.** See §2.

---

## 8. Backbone selection, for later

| Model | ImageNet top-1 | Params | Pixel 6 CPU | Note |
|---|---|---|---|---|
| MobileNetV3-Large (ours) | 75.5% | 5.4M | ~13.6 ms | hard-swish; PTQ-hostile |
| MobileNetV4-Conv-S @224 | 73.8% | 3.8M | 2.4 ms | faster, *less* accurate |
| **MobileNetV4-Conv-M @256** | **79.9%** | 9.2M | 11.4 ms | +4.4 pts at similar CPU latency; conventional ops (ReLU/BN), ONNX-friendly |
| ConvNeXt-Base (ConvNeXtLSD) | — | 87.6M | — | not a browser model |

A backbone change means retraining, re-verifying and re-shipping everything, and its
benefit can only be measured against data we do not yet have. **Defer until the field
set exists.** When it does, MNv4-Conv-M is the candidate.

---

## 9. Prioritised plan — for review; nothing below has been started

| Priority | Item | Measured by | Touches shipped weights? |
|---|---|---|---|
| **P0** | Collect field data through the app: ≥250 vet-labelled photos, cattle **and buffalo** (T3) | the only number that predicts production | no |
| **P1** | Gate: G1 ℓ2-normalised Mahalanobis; G3 KNN/energy comparison; G2 targeted farm-background outliers | same novel set + local photos; retention ≥99% both species | no (lesion path untouched; gate heads only) |
| **P2** | Lesion v1.1: lesion-crop / MIL training from the Roboflow boxes; one Balanced-MixUp arm | locked test split, once: **Mild recall**, specificity ≥0.855 | yes — new artifact, full re-verification |
| **P3** | State PPV at plausible field prevalence in-product and in the card | — | no |
| **P4** | Backbone MNv4-Conv-M | only after P0 | yes |
| — | Not doing: TTA, int8 PTQ, second ImageNet gate, retargeting at 98% | | |

P1 and P2 are independent and can run on Kaggle in parallel. P2 is the one that
addresses what the card calls the model's material weakness; P1 addresses what field
testing exposed. Both are gated on your go-ahead.

---

## 10. Sources read

Read in full or in substantive part:
- Keras, *Transfer learning & fine-tuning* — https://keras.io/guides/transfer_learning/
- Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022 — https://arxiv.org/abs/2202.10054
- Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*, ICLR 2019 — https://arxiv.org/abs/1812.04606
- Mueller & Hein, *Mahalanobis++: Improving OOD Detection via Feature Normalization* — https://arxiv.org/abs/2505.18032
- Szyc et al., *Why OOD detection experiments are not reliable*, UAI 2023 — https://proceedings.mlr.press/v216/szyc23a.html
- Zhang et al., *OpenOOD v1.5* — https://arxiv.org/abs/2306.09301 (abstract and README; leaderboard is JS-rendered and was not read)
- Gessert et al., *Skin lesion classification using CNNs with patch-based attention and diagnosis-guided loss weighting* — https://arxiv.org/abs/1905.02793
- Araújo et al., *Key Patches Are All You Need: MIL for robust medical diagnosis*, CVPR-W 2024 — https://arxiv.org/abs/2405.01654
- Ahmed et al., *Early detection of LSD using deep learning — comparative analysis of pretrained models* — https://pmc.ncbi.nlm.nih.gov/articles/PMC11512320/
- *LSD diagnosis in cattle: MobileNetV2 + RMSProp*, PLOS One 2024 — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0302862
- *Robust multi-class LSD diagnosis for practical livestock applications* (ConvNeXtLSD) — https://pmc.ncbi.nlm.nih.gov/articles/PMC13507278/
- *Benchmarking class-imbalance mitigation across CNNs for skin cancer* — https://pmc.ncbi.nlm.nih.gov/articles/PMC13511939/
- *I Can't Believe TTA Is Not Better: when TTA hurts medical image classification* — https://arxiv.org/html/2604.09697v1
- Smits, *A note on Youden's J and its cost ratio*, BMC Med Res Methodol 2010 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2959030/
- Qin et al., *MobileNetV4*, ECCV 2024 — https://arxiv.org/html/2404.10518
- MobileNetV3 quantisation: TF Model Garden QAT blog; tensorflow/model-optimization #1107 — https://blog.tensorflow.org/2022/06/Adding-Quantization-aware-Training-and-Pruning-to-the-TensorFlow-Model-Garden.html
- Varoquaux & Cheplygina, *ML for medical imaging: methodological failures and recommendations*, npj Digit. Med. 2022 — https://www.nature.com/articles/s41746-022-00592-y (abstract only; full text paywalled from here)

Not read (paywalled/blocked), cited from search summaries only:
- *ViT model-integrated mobile application for LSD detection*, Sci. Rep. 2025 — https://www.nature.com/articles/s41598-025-30259-z
- *Extensive investigation of CNN designs for LSD in dairy cows*, Heliyon 2024 — https://www.sciencedirect.com/science/article/pii/S2405844024102733
