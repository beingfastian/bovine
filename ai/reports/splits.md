# Step 3 — Split Report

**Seed:** 42 · **Folds:** 5 · **Test fraction:** 20%

Stratified on the **joint (label, source)**, not label alone — because source
nearly determines label in this data (`P(lesion|mendeley)=0.299`,
`P(lesion|roboflow)=0.826`). Source-skewed folds would let a model score well by
recognising which dataset an image came from rather than by finding lesions.

## Composition

| Split | Groups | Images | Lesion | Normal | P(lesion) | Sources |
|---|---|---|---|---|---|---|
| test | 559 | 611 | 367 | 192 | 0.657 | mendeley=179 roboflow=380 |
| fold0 | 449 | 476 | 294 | 155 | 0.655 | mendeley=144 roboflow=305 |
| fold1 | 447 | 492 | 292 | 155 | 0.653 | mendeley=143 roboflow=304 |
| fold2 | 445 | 487 | 292 | 153 | 0.656 | mendeley=142 roboflow=303 |
| fold3 | 449 | 485 | 294 | 155 | 0.655 | mendeley=144 roboflow=305 |
| fold4 | 449 | 479 | 294 | 155 | 0.655 | mendeley=144 roboflow=305 |

P(lesion) spread across splits: **0.653 – 0.657**

## Severity coverage (Zenodo labels, joined by SHA-256)

| Split | Normal | Mild | Severe |
|---|---|---|---|
| test | 139 | 47 | 15 |
| fold0 | 111 | 33 | 12 |
| fold1 | 114 | 33 | 18 |
| fold2 | 113 | 39 | 14 |
| fold3 | 108 | 39 | 17 |
| fold4 | 111 | 35 | 14 |

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
