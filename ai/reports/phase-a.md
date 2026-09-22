# Phase A Report — Model Contract Verification

**Date:** 2026-09-20
**Model:** `bovineinsight/assets/finals_models00.tflite`
**Runtime:** `ai-edge-litert` 2.2.0, Python 3.12.10, Windows
**Scripts:** `ai/scripts/inspect_model.py`, `ai/scripts/read_rescaling.py`, `ai/scripts/probe_head.py`

---

## A-T1 — Model contract: **COMPLETE**

| Property | Verified value | Method |
|---|---|---|
| Input tensor | `serving_default_input_layer:0` | interpreter |
| Input shape / dtype | `[1, 224, 224, 3]` float32 | interpreter |
| Input quantization | `(0.0, 0)` — **none, true float32** | interpreter |
| **Input convention** | **raw `[0, 255]`** | **flatbuffer constants (authoritative)** |
| Output tensor | `StatefulPartitionedCall_1:0` | interpreter |
| **Output shape** | **`[1, 1]` — SINGLE value** | interpreter |
| **Output activation** | **`LOGISTIC` (sigmoid) — op 125/125, is the graph output** | flatbuffer op scan |
| Output meaning | **P(class 1 = "Lumpy Skin")**, already a probability in [0,1] | derived |
| Architecture | MobileNetV3-Large | op sequence + layer names |
| Ops used | ADD, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, HARD_SWISH, LOGISTIC, MEAN, MUL, PAD | flatbuffer |
| Total ops | 126 (0–125) | flatbuffer |

### Input convention — proven, not inferred

The first two ops of the graph are the Keras `Rescaling` layer:

```
[0] MUL  input x 0.00784314      (= 1/127.5)
[1] ADD  result + (-1.0)
```

`x / 127.5 - 1` maps **[0, 255] -> [-1, 1]**. This is exactly MobileNetV3's
`include_preprocessing=True` contract.

**Consequence:** feed raw 0-255 pixel values. Do **not** divide by 255. Do **not**
apply ImageNet mean/std. The plan's preprocessing code (section 2.2 step 3) is correct.

### Output head — proven, not inferred

```
[123/125] FULLY_CONNECTED  -> dense_1/MatMul;dense_1/Relu;dense_1/Add
[124/125] FULLY_CONNECTED  -> dense_1_2/MatMul;dense_1_2/Add
[125/125] LOGISTIC         -> StatefulPartitionedCall_1:0   <-- graph output
```

This is a **single-unit binary head with sigmoid**, not a 2-class softmax.

**Consequence:** the output is already a probability. Apply **no** softmax and **no**
sigmoid in client code. `p = output[0][0]` is P(Lumpy Skin); `1 - p` is P(Normal Skin).

### Assertion results

| Check | Result |
|---|---|
| input is 4-D NHWC | PASS |
| input is 224x224 | PASS |
| input has 3 channels | PASS |
| input dtype float32 | PASS |
| output classes == labels.txt count | **FAIL — model outputs 1, labels.txt lists 2** |

The final FAIL is **expected and correct**: a sigmoid binary head has one output for two
classes. `labels.txt` is not wrong, but it cannot be used to size the output tensor.
This is the source of new bug **A6** (below).

---

## A-T5 — Synthetic / degenerate input probe: **COMPLETE (inconclusive by design)**

| Input | Output |
|---|---|
| all zeros (black) | 0.000000 |
| all 255 (white) | 0.000000 |
| uniform random 0-255 | 0.000000 |
| uniform random 0-1 | 0.981371 |
| gaussian mean 128 | 0.000000 |
| mid grey 128 | 0.000000 |

**Do not read a preprocessing conclusion from this table.** An early heuristic in
`probe_head.py` concluded from the observed `[0, 0.98]` range that inputs should be
`[0,1]`-normalised. **That conclusion was wrong** — it was an artifact of feeding
degenerate non-images. The `[0,1]` case maps to `[-1, -0.992]` after rescaling, i.e. a
near-constant tensor, and 0.98 is what the network happens to emit for it. The
flatbuffer constants above are authoritative and say `[0,255]`.

**What this table IS useful for:** it confirms the section 6.1 premise. The model returns
a confident, saturated answer for black frames, white frames and pure noise. It has no
capacity to abstain. **Gates 1-3 are load-bearing, not polish.**

Real-image A-T5 (wall / person / goat photos) still pending.

---

---

## A-T4 — Accuracy on Mendeley 1,024: **COMPLETE — FAIL**

**Dataset:** Mendeley `w36hpf86j2` v1, SHA-256 verified `8f683d7b…2331`, 1,024 PNGs
(324 `Lumpy Skin/` + 700 `Normal Skin/`), 0 unreadable. Stored at
`ai/data/lumpy_skin_dataset.zip`. CC BY 4.0 — Kumar, S. & Shastri, S. (2022).

### Result with the VERIFIED contract (raw `[0,255]`)

| Metric | Value |
|---|---|
| Distinct output values across 1,024 images | **1** |
| Output min / max / mean | **0.0000 / 0.0000 / 0.0000** |
| ROC-AUC | **0.5000** (as-is and inverted) |
| Accuracy @0.50 | 0.6836 — **exactly the class prior** (700/1024) |
| TP / TN / FP / FN | 0 / 700 / 0 / 324 |

**The model emits a constant 0.0 for every single image.** It predicts "Normal" for
everything, including all 324 diseased animals. Zero discriminative power.

### Was it our preprocessing? — No. Convention sweep (300-image stratified subset)

| Convention | ROC-AUC | distinct outputs |
|---|---|---|
| **raw `[0,255]`** (the graph's own contract) | 0.5000 | **1** |
| `/255` → `[0,1]` | 0.3368 | 297 |
| `/127.5 - 1` → `[-1,1]` | 0.4886 | 238 |
| BGR raw `[0,255]` | 0.5000 | **1** |
| BGR `/255` | 0.3342 | 296 |
| Caffe mean-subtraction | 0.5000 | **1** |

### Full-dataset confirmation on the best convention (`/255`, n=1,024)

| Metric | Value |
|---|---|
| ROC-AUC as-is | 0.3249 |
| **ROC-AUC inverted** | **0.6751** |
| Polarity | **INVERTED** — output tracks P(Normal), not P(Lumpy) |
| **Best balanced accuracy** | **0.6253** @ threshold 0.32 |
| Accuracy at that threshold | 0.6191 |
| Recall / Specificity | 0.6420 / 0.6086 |
| TP / TN / FP / FN | 208 / 426 / 274 / 116 |
| **Majority-class baseline** | **0.6836** |

**The model's best achievable accuracy (0.6191) is below the "always say normal"
baseline (0.6836).** A constant function outperforms it.

### Diagnosis — a training-time preprocessing bug, baked into the weights

The evidence is consistent and points one way:

1. The graph's `Rescaling` layer maps `[0,255] → [-1,1]`.
2. Fed that correct range, the network outputs a **constant 0.0** — activations have gone
   so far outside the range the weights ever saw that the penultimate ReLU dies.
3. Fed `[0,1]` instead, rescaling produces `[-1, -0.992]` — a razor-thin, nearly constant
   band — and the model produces *varied* output with weak signal (AUC 0.675).

**The model was almost certainly trained on `[0,1]`-normalised images fed into a network
that already contained `include_preprocessing=True`.** The training data was squeezed into
a degenerate 0.8%-wide slice of the input space. The weights learned a little noise there
and nothing generalisable.

This is not an inference bug we can fix. **The weights encode the bug.**

### A-T4 DECISION GATE (plan §2.4): **FAIL** — 0.6253 vs 0.80 required

**Phase B is a full retrain.** The existing `.tflite` is retained only as a historical
artifact. It must not ship in any form, including as a fallback.

### What survives

- **MobileNetV3-Large was the right architecture choice** — keep it for the retrain.
- The **input contract** (`[0,255]`) and **output contract** (1-unit sigmoid) are now
  documented and verified, so the Dart/ONNX client code is written against a known spec.
- The **retrain must not repeat the bug**: either keep `include_preprocessing=True` and
  feed raw `[0,255]` throughout training *and* inference, or set it `False` and normalise
  explicitly. Never both. Add an assertion on the training input range.

---

## Outstanding

| Test | Status | Note |
|---|---|---|
| A-T2 — Python↔Dart parity | **Deferred** | Pointless against dead weights. Revive after Phase B produces a real model |
| A-T3 — determinism | **Deferred** | Same |
| A-T5 — real-image OOD probe | Partial | Re-run against the Phase B model |

**Phase A is complete.** Its purpose was to determine whether the existing weights were
usable. They are not. Phase B begins as a full retrain, not a refinement.
