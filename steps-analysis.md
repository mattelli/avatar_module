# steps-analysis.md — Talking-head pipeline: step-by-step analysis

**Status: implemented (Plan 7, tasks 1–6; pose warp removed).**

This document analyses the **agreed pipeline**:

```
src ──▶ [0] square image (N×N)
    ──▶ [1] tighter detected-bbox crop_face M×M (M<N), resize M×M → 96×96
    ──▶ [2] Wav2Lip(clean)
    ──▶ [3] resize (96×96 → 112×112), expression warp (PFLD)
    ──▶ [4] (112×112 → M×M) and in-paint into the square image (N×N)
    ──▶ [5] resize → frame
```

For each step: **input**, **what** it does, **how** it does it (algorithm +
constants + code refs), **output**.

File references are to `crates/avatar-core/src/` unless noted.

---

## Decisions (confirmed)

| # | Decision |
|---|----------|
| 1 | **Landmark model: PFLD** — convert `pfld.onnx` (33 KB) → `.bpk`, place in `.res/models/pfld/`, load like Wav2Lip/ExpNet |
| 2 | PFLD-pytorch checkpoint **license: not a concern** (per user) |
| 3 | **Landmarks detected on the clean face**; **expression warp runs on the step-1 crop** (before the in-paint) |
| 4 | **ExpNet → displacement mapping stays hand-tuned** (confirmed) |
| 5 | **Wav2Lip framing: tighter detected-bbox crop** (`crop_face`), not the whole square |
| 6 | **Per-frame Wav2Lip: one clean face per chunk, reused across frames** (see step 2) |
| 7 | The 96→size upscale blur is **accepted** |
| 8 | **In-paint boundary ramp = 2 px** (step 3) |
| 9 | **N = the input image's square size** (e.g. N = 740 for `avatar.jpg`); no face crop at step 0 |
| 10 | **Step 3 (expression warp) runs at 112×112** (PFLD's resolution) |

---

## Why this pipeline (context)

The previous pipeline had two defects:

1. **Seam** — Wav2Lip ran on the *unwarped* `src` while its mouth was blended into
   the *warped* frame through a **fixed** ellipse → visible patch seam.
2. **Input-dependent geometry** — the 9 expression control points and the mouth
   mask were hardcoded canonical positions, not detected.

This pipeline fixes both: Wav2Lip runs on a **tightly cropped clean face**, its
output is **pasted back** (no mask/blend of a sub-region), and the expression
warp uses **PFLD-detected** control points.

---

## Step 0 — Square image (N×N)

**Where:** `talking_head.rs::load_source_face` (adapted).

| | |
|---|---|
| **Input** | avatar picture (PNG/JPEG) or `None` |
| **What** | make the input square at its **native resolution N×N** (no face crop here) |
| **How** | `load_image` → `atlas::center_crop_square` (crops the longer axis, centered, if the input is not square). If no picture: `atlas::test_face_cpu(N, N)`. |
| **Output** | `sq` — RGBA **N×N** |

`N` is the **input image's** square size — e.g. **N = 740** for `avatar.jpg`
(740×740). For a non-square input, `N = min(width, height)` after the centered
square crop. **No face crop happens here** (that is step 1); this preserves the
full frame context (background + face) for the later in-paint.

---

## Step 1 — Tighter bbox crop → 96×96

**Where:** `wav2lip::crop_face` (lines 302–318) + `wav2lip::face_input` (270–298).

| | |
|---|---|
| **Input** | `sq` (N×N) |
| **What** | crop the **detected face bbox** (tighter than the square) to **M×M** (M<N), then resize to Wav2Lip's native **96×96** |
| **How** | ① `face_crop`/`detect_face_box` → bbox; ② `wav2lip::crop_face(img, bbox)` applies the Wav2Lip **pads** `[0,10,0,0]` (pady1, pady2, padx1, padx2) and resizes the crop to 96×96; ③ `face_input` builds the 6-channel tensor `[masked_face (rows 48..96 zeroed), full_face]`, `0..1` → `[6,96,96]` |
| **Output** | `face96` — `[6,96,96]` tensor; **plus** the bbox (needed by step 3) |

Notes: the tighter crop matches Wav2Lip's training framing (face fills the
frame), improving lip quality vs the whole square.

---

## Step 2 — Wav2Lip(clean)

**Where:** `wav2lip::Wav2Lip::predict` (lines 77–135) → Burn
`crates/wav2lip/src/model/wav2lip.rs::forward`.

| | |
|---|---|
| **Input** | `face96` (same for every frame) + mel chunk `[1,80,16]` |
| **What** | re-render the face so the **mouth matches the speech** (lip-sync GAN); outputs a **full face** |
| **How** | `Model::forward`: audio encoder `[b,1,80,16]→[b,512,1,1]`; face encoder → pyramid; decoder + skip connections → `[b,3,96,96]`, sigmoid → `0..1` |
| **Output** | `w2l` — `[3,96,96]` (channel-major) full face, `0..1` |

### Batching strategy (confirmed)
The clean face is **identical for every frame**, so only the mel chunk varies.
Wav2Lip therefore runs **once per mel chunk** (chunks ≈ frames), batching chunks
(default **8**), with the **same face tensor reused** across all chunks. This is
the current behaviour and is kept.

---

## Step 3 — Resize 96×96 → 112×112, expression warp (PFLD)

**Where:** `fusion.rs::warp_from_landmarks` (replaces `warp_from_coeffs`), plus
the **PFLD** crate (`crates/pfld`).

| | |
|---|---|
| **Input** | `w2l` (`[3,96,96]`); `exp_coeffs[64]` (ExpNet) |
| **What** | resize the Wav2Lip face to **112×112** (PFLD's native input), detect landmarks, and deform brows/eyes/mouth-corners/chin to show **non-lip expression** |
| **How** | ① `patch_to_image(w2l)` → RGBA 96×96 (channel-major — see the CHW fix); ② `resize(…, 112, 112, Triangle)`; ③ **PFLD**: RGB, normalize `0..1`, CHW → `[1,3,112,112]` → `landmarks [1,196]` → reshape to **98 × (x,y)**, normalized `0..1`; ④ map 98 → **9 control points**; ⑤ set each point's **displacement** from the ExpNet coefficients (as today: `coeff[0]`→mouth open, `coeff[3]`→brow, `coeff[5/6]`→eyes, `coeff[7]`→chin), clamped `[-1,1]`; ⑥ `atlas::warp_cpu` inverse-distance warp: per output pixel `disp = Σ(w·cp.disp)/Σw`, `w = 1/(d²+0.004)`; sample source at `pn − disp·strength`, bilinear |
| **Output** | `expr` — RGBA **112×112** |

> **Why the face is cropped first (design fix).** PFLD needs a centered,
> frame-filling face. Running it on the full N×N square (face ≈ 48 % of the
> width) makes the landmarks scatter (measured: only 64 % of landmarks land
> inside the detector's face bbox, chin 161 px too low). Running PFLD on the
> step-1 crop puts **96 %** of landmarks on the face (chin within 3 px). That is
> why the expression warp (step 3) runs **before** the in-paint (step 4) — the
> warp sees the clean cropped face, and its output is pasted back afterwards.

### 98 → 9 mapping (WFLW layout)

| Control point | From PFLD indices |
|---|---|
| mouth center | mean of inner mouth 89–95 |
| mouth left | outer mouth corner 76 |
| mouth right | outer mouth corner 82 |
| brow left | mean of 33–41 |
| brow right | mean of 42–50 |
| eye left | pupil 96 |
| eye right | pupil 97 |
| nose | tip 57 (never displaced) |
| chin | lowest contour point 15 |

Notes:
- **Landmarks fix positions, not displacements.** PFLD says *where* the features
  are; the ExpNet→displacement mapping still decides *how much* to move. That
  mapping is **unchanged and hand-tuned** (decision 4).
- PFLD outputs **two** tensors (`out1 [1,64,28,28]` + `landmarks [1,196]`); only
  `landmarks` is used.
- PFLD coordinates are **normalized**, so they apply directly to the 112×112
  frame — no coordinate round-trip.
- Landmarks are detected on the **clean** (unwarped) face — decision 3.
- The expression warp deforms the **whole** 112×112 face, **including the
  Wav2Lip mouth** (the mouth control points are not excluded) — user-confirmed.

---

## Step 4 — 112×112 → M×M, in-paint into N×N

**Where:** `inpaint.rs::crop_face_padded` (step 1) + `inpaint.rs::paste_face`.

| | |
|---|---|
| **Input** | `expr` (112×112); the bbox rect from step 1; `sq` (N×N) |
| **What** | scale the warped face back to **M×M** and paste it into the N×N square at the bbox location, blending the boundary so there is no visible seam |
| **How** | ① `resize(expr, M, M, Triangle)`; ② paste the M×M patch into `sq` at the bbox rect; ③ **feather** the boundary with a **2-pixel** alpha ramp so the paste edge is invisible |
| **Output** | `sq2` — RGBA **N×N**: the original square with the re-rendered face pasted in |

Notes: this is the "in-paint" — it re-inserts the Wav2Lip face into the full
frame. No sub-region mouth mask is used; the whole cropped face is replaced. The
boundary ramp is **2 px** (decision). The paste is written as a **region table**
(`InpaintRegion`) so a future region (e.g. an eye patch for blinking) can be
added without reworking the pipeline.

---

## Step 5 — Resize → frame

**Where:** `talking_head.rs::render_talking_head` (final resize).

| | |
|---|---|
| **Input** | `sq2` (N×N); `size` ∈ {64,128,256,512} |
| **What** | scale to the requested output size |
| **How** | `imageops::resize(…, size, size, Triangle)` (optionally `CatmullRom`/`Lanczos3`) |
| **Output** | RGBA **size×size** — the final video frame |

Notes: N→512 is a downscale for `avatar.jpg` (N=740); for a small source it may
be an upscale and is inherently soft (**accepted**). "Smooth" = the interpolation
filter; it cannot recover missing detail.

> **Pose warp removed.** Earlier versions applied an audio-driven 2D affine
> head-pose warp (Audio2Pose → `fusion::pose_warp`) on the whole N×N frame
> between the in-paint and this resize. It was removed: the render now goes
> straight from the in-paint (step 4) to the resize. The Audio2Pose model, the
> Wav2Lip audio-encoder embedding API, the `pose` crate and the `pose_seed`
> knob were removed with it.

---

## Data-flow summary

| Step | Reads | Writes | Resolution |
|---|---|---|---|
| 0 | picture | `sq` | N×N |
| 1 | `sq` + bbox | `face96` (+ bbox) | 96×96 |
| 2 | `face96` + mel chunk | `w2l` | 96×96 |
| 3 | `w2l` + coeffs | `expr` (+ PFLD landmarks) | 112×112 |
| 4 | `expr` + bbox + `sq` | `sq2` | N×N |
| 5 | `sq2` + size | frame | size×size |

---

## Current vs new

**Current:**
```
src ──▶ expression warp ──▶ pose warp ──▶ [blend Wav2Lip(src) mouth] ──▶ frame
```
- Wav2Lip on unwarped `src`; fixed mask; misaligned → **seam**.
- Hardcoded control points.

**New:**
```
src ─▶ [0] square N×N ─▶ [1] bbox crop M×M → 96×96 ─▶ [2] Wav2Lip(clean)
    ─▶ [3] 96×96 → 112×112, expression warp (PFLD) ─▶ [4] 112×112 → M×M, in-paint into N×N
    ─▶ [5] resize ─▶ frame
```
- Wav2Lip on a **tightly cropped clean face** → best lips.
- Whole face pasted back (no sub-region mask) → **no seam**.
- Control points **detected** (PFLD) → not input-agnostic.
- No head-pose warp (the Audio2Pose step was removed).

---

## Resolved open items

| Item | Resolution |
|---|---|
| PFLD → Burn | Convert `pfld.onnx` (33 KB) → `.bpk`; place in `.res/models/pfld/`; load like Wav2Lip/ExpNet |
| PFLD license | Not a concern (user) |
| Detection face | Clean (unwarped) face; expression warp runs on the step-1 crop (before the in-paint) |
| ExpNet→displacement mapping | Stays hand-tuned |
| Wav2Lip framing | Tighter detected-bbox crop (`crop_face`) |
| Per-frame Wav2Lip | One clean face per chunk, reused across frames; batch chunks (default 8) |

---

## Remaining implementation notes

1. **PFLD I/O (verified):** input `input_1 [1,3,112,112]` float `0..1`; outputs
   `out1 [1,64,28,28]` and `landmarks [1,196]` (98 × (x,y), normalized). Use only
   `landmarks`.
2. **PFLD needs a centered, cropped face** (verified: landmarks scatter on an
   uncropped, off-center face) — step 1's crop is a prerequisite. **Resolved** by
   running the expression warp (step 3) on the step-1 crop, *before* the
   in-paint (step 4). Measured on `avatar.jpg`: landmarks inside the detector
   bbox 64 % (full square) → **96 %** (crop); chin error 161 px → **3 px**.
3. **N and M** — N = the input square size (e.g. 740); M = the detected bbox
   size. Resize chain: 96 (Wav2Lip) → 112 (PFLD/warp) → M (in-paint) → N (square)
   → size (output).
4. **Performance:** step 5 downscales the N×N square (e.g. 740×740) to `size`.
   N is kept **native** (user-confirmed); consider capping at `min(N, 512)` if
   the in-paint proves slow.
