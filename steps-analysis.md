# steps-analysis.md — Talking-head pipeline: step-by-step analysis

This document analyses the **agreed pipeline**:

```
src ──▶ [0] square image (N×N)
    ──▶ [1] tighter detected-bbox crop_face M×M (M<N), resize M×M → 96×96
    ──▶ [2] Wav2Lip(clean)
    ──▶ [3] (96×96 → M×M) and in-paint into the square image (N×N)
    ──▶ [4] resize (N×N → 112×112), expression warp (PFLD)
    ──▶ [5] pose warp
    ──▶ [6] resize → frame
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
| 3 | **Landmarks detected on the clean face**; **expression warp before pose warp** |
| 4 | **ExpNet → displacement mapping stays hand-tuned** (confirmed) |
| 5 | **Wav2Lip framing: tighter detected-bbox crop** (`crop_face`), not the whole square |
| 6 | **Per-frame Wav2Lip: one clean face per chunk, reused across frames** (see step 2) |
| 7 | The 96→size upscale blur is **accepted** |
| 8 | **In-paint boundary ramp = 2 px** (step 3) |
| 9 | **N = the input image's square size** (e.g. N = 740 for `avatar.jpg`); no face crop at step 0 |
| 10 | **Steps 4–5 run at 112×112** (PFLD's resolution) |

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

## Step 3 — 96×96 → M×M, in-paint into N×N

**Where:** new (uses `fusion::patch_to_image` + `imageops::resize` + a feathered paste).

| | |
|---|---|
| **Input** | `w2l` (`[3,96,96]`); the bbox from step 1; `sq` (N×N) |
| **What** | scale the Wav2Lip face back to **M×M** and paste it into the N×N square at the bbox location, blending the boundary so there is no visible seam |
| **How** | ① `patch_to_image(w2l)` → RGBA 96×96 (channel-major — see the CHW fix); ② `resize(…, M, M, Triangle)`; ③ copy the M×M patch into `sq` at the bbox rect; ④ **feather** the boundary with a **2-pixel** alpha ramp so the paste edge is invisible |
| **Output** | `sq2` — RGBA **N×N**: the original square with the re-rendered face pasted in |

Notes: this is the "in-paint" — it re-inserts the Wav2Lip face into the full
frame. No sub-region mouth mask is used; the whole cropped face is replaced. The
boundary ramp is **2 px** (decision).

---

## Step 4 — Resize N×N → 112×112, expression warp (PFLD)

**Where:** new — `fusion.rs::warp_from_landmarks` (replaces `warp_from_coeffs`),
plus a new **PFLD** module (e.g. `crates/pfld` or `avatar-core::landmarks`).

| | |
|---|---|
| **Input** | `sq2` (N×N); `exp_coeffs[64]` (ExpNet) |
| **What** | resize to **112×112** (PFLD's native input), detect landmarks, and deform brows/eyes/mouth-corners/chin to show **non-lip expression** |
| **How** | ① `resize(sq2, 112, 112, Triangle)`; ② **PFLD**: RGB, normalize `0..1`, CHW → `[1,3,112,112]` → `landmarks [1,196]` → reshape to **98 × (x,y)**, normalized `0..1`; ③ map 98 → **9 control points**; ④ set each point's **displacement** from the ExpNet coefficients (as today: `coeff[0]`→mouth open, `coeff[3]`→brow, `coeff[5/6]`→eyes, `coeff[7]`→chin), clamped `[-1,1]`; ⑤ `atlas::warp_cpu` inverse-distance warp: per output pixel `disp = Σ(w·cp.disp)/Σw`, `w = 1/(d²+0.004)`; sample source at `pn − disp·strength`, bilinear |
| **Output** | `expr` — RGBA **112×112** |

### 98 → 9 mapping (WFLW layout)

| Control point | From PFLD indices |
|---|---|
| mouth center | mean of inner mouth 76–95 |
| mouth left | outer mouth corner 68 |
| mouth right | outer mouth corner 75 |
| brow left | mean of 33–37 |
| brow right | mean of 38–42 |
| eye left | mean of 52–59 (or pupil 96) |
| eye right | mean of 60–67 (or pupil 97) |
| nose | tip 51 (never displaced) |
| chin | lowest contour point (e.g. 16) |

Notes:
- **Landmarks fix positions, not displacements.** PFLD says *where* the features
  are; the ExpNet→displacement mapping still decides *how much* to move. That
  mapping is **unchanged and hand-tuned** (decision 4).
- PFLD outputs **two** tensors (`out1 [1,64,28,28]` + `landmarks [1,196]`); only
  `landmarks` is used.
- PFLD coordinates are **normalized**, so they apply directly to the 112×112
  frame — no coordinate round-trip.
- Landmarks are detected on the **clean** (unwarped) face — decision 3.

---

## Step 5 — Pose warp

**Where:** `fusion.rs::pose_warp` (lines 355–416).

| | |
|---|---|
| **Input** | `expr` (112×112); `pose[6]` = `[yaw,pitch,roll,tx,ty,tz]` (Audio2Pose); `strength` |
| **What** | apply **audio-driven head motion** — rotation, slight shear, scale, translation — as one 2D affine transform about the frame centre |
| **How** | clamp each component to `[-1,1]`; parameters (× `strength`): `ang = roll·0.15`; `shear_x = yaw·0.20`; `shear_y = pitch·0.20`; `off_x = tx·0.10·w`, `off_y = ty·0.10·h`; `scale = 1 + tz·0.06`. Per output pixel: coords about centre → **inverse** transform (rotate→scale→shear→translate) → clamp to edges → bilinear sample |
| **Output** | `posed` — RGBA **112×112** |

Notes: 2D affine approximation of 3D head pose. Edge clamping → bounded, no NaN.
Because the expression warp already ran (step 4), the head pose is applied to the
**whole** frame — mouth included — so there is **no seam**.

---

## Step 6 — Resize → frame

**Where:** new (analogous to the final `resize` in `fuse_frame`, lines 441–448).

| | |
|---|---|
| **Input** | `posed` (112×112); `size` ∈ {64,128,256,512} |
| **What** | scale to the requested output size |
| **How** | `imageops::resize(…, size, size, Triangle)` (optionally `CatmullRom`/`Lanczos3`) |
| **Output** | RGBA **size×size** — the final video frame |

Notes: 112→512 is a 4.6× upscale and is inherently soft (**accepted**). 64/128
are downscales — fine. "Smooth" = the interpolation filter; it cannot recover
missing detail.

---

## Data-flow summary

| Step | Reads | Writes | Resolution |
|---|---|---|---|
| 0 | picture | `sq` | N×N |
| 1 | `sq` + bbox | `face96` (+ bbox) | 96×96 |
| 2 | `face96` + mel chunk | `w2l` | 96×96 |
| 3 | `w2l` + bbox + `sq` | `sq2` | N×N |
| 4 | `sq2` + coeffs | `expr` (+ PFLD landmarks) | 112×112 |
| 5 | `expr` + pose | `posed` | 112×112 |
| 6 | `posed` + size | frame | size×size |

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
    ─▶ [3] 96×96 → M×M, in-paint into N×N ─▶ [4] N×N → 112×112, expression warp (PFLD)
    ─▶ [5] pose warp ─▶ [6] resize ─▶ frame
```
- Wav2Lip on a **tightly cropped clean face** → best lips.
- Whole face pasted back (no sub-region mask) → **no seam**.
- Control points **detected** (PFLD) → not input-agnostic.

---

## Resolved open items

| Item | Resolution |
|---|---|
| PFLD → Burn | Convert `pfld.onnx` (33 KB) → `.bpk`; place in `.res/models/pfld/`; load like Wav2Lip/ExpNet |
| PFLD license | Not a concern (user) |
| Detection face | Clean (unwarped) face; expression warp before pose warp |
| ExpNet→displacement mapping | Stays hand-tuned |
| Wav2Lip framing | Tighter detected-bbox crop (`crop_face`) |
| Per-frame Wav2Lip | One clean face per chunk, reused across frames; batch chunks (default 8) |

---

## Remaining implementation notes

1. **PFLD I/O (verified):** input `input_1 [1,3,112,112]` float `0..1`; outputs
   `out1 [1,64,28,28]` and `landmarks [1,196]` (98 × (x,y), normalized). Use only
   `landmarks`.
2. **PFLD needs a centered, cropped face** (verified: landmarks scatter on an
   uncropped, off-center face) — step 1's crop is a prerequisite.
3. **N and M** — N = the input square size (e.g. 740); M = the detected bbox
   size. Confirm the resize chain (96→M→N→112).
4. **Performance:** step 3 pastes into an N×N frame (e.g. 740×740) and step 4
   resizes N×N→112×112 — the large intermediate may be costly; consider working
   at a capped N (e.g. min(N, 512)) if needed.
