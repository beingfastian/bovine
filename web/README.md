# BovineInsight — web

A single-page screening app. Camera or gallery in, three-way verdict out, with
the model running **in the browser** via onnxruntime-web. No inference server,
no cold starts, static hosting only.

---

## What is running

| | |
|---|---|
| Model | `bi-lsd-mnv3l-v1.0.0-gate`, ONNX opset 13, **float16 weights / fp32 I/O**. Lesion weights are `v1.0.0` byte-for-byte; `-gate` adds two heads on the same backbone |
| Download | ~6.3 MB (~5.7 MB gzipped) model + 14 MB (3.6 MB gzipped) WASM runtime, both cached for a year |
| Inference | ~50-60 ms median on desktop Chrome, single-threaded WASM |
| Input | raw `[0,255]` RGB, `1x224x224x3` NHWC float32 |
| Output `lesion` | `[1,1]` float32, **already a sigmoid probability** |
| Output `species` | `[1,3]` softmax over cattle / buffalo / other |
| Output `ood` | `[1,1]` Mahalanobis d² of the pooled features from the bovine training distribution |
| Threshold | τ = 0.50, with `[0.40, 0.60)` routed to "unclear" |
| Gate 1 | reject when `ood ≥ oodMax` **or** `p(other) ≥ otherMax`; both from `thresholds.json` |

Outputs are matched by **name**, not position or shape — two of them are `[1,1]`.

### The input contract, once more

The graph rescales internally (`MUL 1/127.5`, `ADD -1`). **Do not** divide by
255. **Do not** apply ImageNet mean/std. **Do not** apply a second sigmoid. A
mismatch of exactly this kind destroyed the predecessor model — see
`MODEL_CARD.md` §7. `lib/model.ts` and `lib/preprocess.ts` carry the same
warning where the code is.

---

## Setup

```bash
cd web
npm install                 # also copies the WASM runtime into public/ort/<version>/
cp .env.local.example .env.local   # then fill in the two Supabase values
npm run dev                 # http://localhost:3000
```

Without `.env.local` everything works except sending: the app scores photos and
shows verdicts, and the submit panel says logging is not configured.

### Supabase (one-time)

1. Create a free project at supabase.com — no card required.
2. SQL Editor → New query → paste **all** of `../supabase/schema.sql` → Run.
   That creates `screenings`, `screening_labels`, the RLS policies, the
   private `screenings` storage bucket and the two views `/review` reads.
   **Existing project?** Run `../supabase/migrations/002_gate_species_autosave.sql`
   instead — it is idempotent and also removes the automated test rows.
3. Project Settings → Data API → copy the URL.
   Project Settings → API Keys → copy the **anon / publishable** key.
4. Put both in `.env.local`.

Both values are compiled into the client bundle and are meant to be public.
What protects the data is the RLS policy, not the key: anonymous visitors can
**insert** a screening and upload one photo, and can **read nothing**.

**The `service_role` key must never appear in this directory.**

To use `/review`, sign in with a magic link. Supabase sends those to any email
by default; restrict it under Authentication → Providers if you want only your
own address to work.

---

## Deploying

Static export, so any static host works.

**Cloudflare Pages** — connect the repo, then:
- Build command: `npm run build`
- Build output directory: `web/out`
- Root directory: `web`
- Environment variables: the two `NEXT_PUBLIC_SUPABASE_*` values

**Netlify** — `netlify.toml` already has it.

`public/_headers` applies on both, and is doing real work: it caches the model
and the WASM runtime as `immutable` for a year (both live at versioned paths),
while keeping `thresholds.json` on a 5-minute TTL so the operating point can be
retuned by editing one file.

Not Vercel Hobby — its terms prohibit commercial use.

---

## Testing

Two harnesses, both driving the real Chrome already installed on the machine
via `puppeteer-core` (no 130 MB browser download).

```bash
npm run build
npm run serve          # in one terminal
npm run test:parity    # in another
npm run test:e2e
```

### `test:parity` — the numbers

Scores 12 fixtures in the browser and compares against probabilities computed
in Python by the verified ONNX model. This is not ceremony: the first time it
ran it **caught a real bug**. Canvas `drawImage()` downscaling is not PIL
bilinear, drift reached 0.027, and a verdict flipped. `lib/resize.ts` is a port
of Pillow's filter written to close that gap, and it took max drift down to
0.005.

The same page is served at `/selftest`. Open it on any phone before using that
phone for collection — it is the only way to know that device's JPEG decoder
and canvas agree with the reference.

Current: max drift **0.0073**, mean **0.0018**, zero non-boundary verdict flips.

One fixture sits ~0.002 from the 0.60 dead-band edge and flips under any
nonzero drift. It is reported separately rather than counted as a failure,
because counting it would amount to demanding bit-exact equality with Pillow.

### `test:e2e` — the screen

Walks the real page: picks a file, checks the disclosure is visible and there
is no Send button, checks the photo is saved **before** any answer, checks
species was pre-filled by the detector, answers the chips, checks the buffalo
warning fires, reads the raw score out of the technical panel and confirms it
matches Python, confirms the answer was saved — and asserts the forbidden copy
never renders. A third leg feeds a real landscape photograph (a Windows
wallpaper, if the machine has one) and asserts it is turned away with no
lesion verdict shown, and still saved.

`ALLOWED` in `scripts/e2e-page.mjs` holds exactly one sanctioned phrase ("not a
diagnosis", which the safety rules *require*). Every other appearance of
"healthy", "lumpy skin", "LSD" or "diagnosis" fails the run. Sanctioning a new
one has to be a deliberate edit to that list.

---

## Things worth knowing before changing something

**The label is asked before the verdict is shown.** `ASK_BEFORE_REVEAL` in
`app/page.tsx`. Someone who has just read "possible skin condition" is no
longer an independent observer, and independent field labels — especially on
buffalo — are the entire point of the app. Set it to `false` to reveal
immediately.

**Gate 1 has two halves, and the second exists because the first failed a real
test.** The obvious design — a cattle/buffalo/other softmax head — called every
one of twenty landscape photographs "cattle" with p = 1.000. The fine-tuned
backbone learned that fields, grass and sky mean cattle, and a closed softmax
cannot say "none of these". So the graph also emits a Mahalanobis distance from
the bovine training distribution (`ood`), which is open-set: it does not need
to have seen the category. The softmax's `p(other)` is kept as a cheap second
opinion for the negative categories it *was* shown. Held-out: species
cattle→cattle 99.7%, buffalo→buffalo 100%; OOD AUROC 0.956; on 6,588
non-animal photos from categories never trained on, the shipped gate rejects
**78%** (distance alone 29%, softmax alone ~75% — the softmax carries most of
it once its negatives are broad; the distance term is what catches
high-resolution landscapes), while keeping 99.0% of held-out cattle and 100%
of buffalo. The six
cattle photos it does turn away are itemised in
`../ai/reports/gate_rejected_cattle_inspection.json` — four are dataset
contamination (a screenshot, a cartoon, a statue, an antelope). Full numbers in
`../ai/reports/gate_report.json` and `../ai/reports/gate_verification.json`.
Gate 2 (blur and exposure) is in `lib/quality.ts`.

**Species is pre-filled from the detector, not asked cold.** The person taps
only if it is wrong, and whether they changed it is recorded
(`screening_labels.species_changed`) — a high override rate means the detector
is wrong, or the question is unclear.

**Saving is automatic and the screen says so.** The photo and every model
output are written the instant a verdict exists, before any answer; the
person's answers land in `screening_labels` afterwards as a separate insert,
because anonymous visitors have no UPDATE right anywhere. There is no Send
button. A permanent line under the header states that photos are saved and
that no name, phone number or location is.

**Every screening is saved, including "unclear" and "not an animal".** A
rising abstention rate is the earliest signal of drift, and a wrongly rejected
real animal is the worst outcome for a data-collection app — so the rejection
screen asks "is there actually a cattle or buffalo here?" and turns that photo
into a labelled example of the gate being wrong instead of a lost one.

**`thresholds.json` is config, not code.** Every number in it was measured;
the file says which sweep produced each one.

**int8 quantisation was tried and rejected.** 2.9 MB instead of 5.7 MB, but
decision agreement fell to 0.81 and balanced accuracy to 0.70. Numbers in
`ai/reports/quantization.json`. Do not re-attempt it without re-running that
script.
