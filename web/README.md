# BovineInsight — web

A single-page screening app. Camera or gallery in, three-way verdict out, with
the model running **in the browser** via onnxruntime-web. No inference server,
no cold starts, static hosting only.

---

## What is running

| | |
|---|---|
| Model | `bi-lsd-mnv3l-v1.0.0`, ONNX opset 13, **float16 weights / fp32 I/O** |
| Download | 6.3 MB (5.7 MB gzipped) model + 14 MB (3.6 MB gzipped) WASM runtime, both cached for a year |
| Inference | ~50-60 ms median on desktop Chrome, single-threaded WASM |
| Input | raw `[0,255]` RGB, `1x224x224x3` NHWC float32 |
| Output | `[1,1]` float32, **already a sigmoid probability** |
| Threshold | τ = 0.50, with `[0.40, 0.60)` routed to "unclear" |

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
   That creates the `screenings` table, the RLS policies, the private
   `screenings` storage bucket and the `collection_summary` view.
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

Walks the real page: picks a file, checks the label step gates the reveal,
answers both chips, checks the buffalo warning fires, reads the raw score out
of the technical panel and confirms it matches Python — and asserts the
forbidden copy never renders.

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

**Gate 1 is not implemented.** There is no animal detector, so the app will
happily score a photo of a wall. `LimitsBanner` says so out loud. Gate 2 (blur
and exposure) is in `lib/quality.ts`.

**Every screening is submitted, including "unclear" ones.** A rising abstention
rate is the earliest signal that the model is drifting away from the field
distribution.

**`thresholds.json` is config, not code.** Every number in it was measured;
the file says which sweep produced each one.

**int8 quantisation was tried and rejected.** 2.9 MB instead of 5.7 MB, but
decision agreement fell to 0.81 and balanced accuracy to 0.70. Numbers in
`ai/reports/quantization.json`. Do not re-attempt it without re-running that
script.
