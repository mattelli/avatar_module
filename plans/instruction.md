# Instruction — AI-Agent: Diffusion I2I Avatar Asset Generation

**Goal:** Given a single avatar reference image, an AI agent uses an image-to-image
(I2I) diffusion pipeline to generate a **consistent set of avatar assets** (expressions,
lip shapes, head poses, eyes) that the `avatar_module` pipeline can interleave at
runtime to produce a talking avatar video.

**This is a pre-processing / asset-generation step.** It runs **once per avatar**,
offline. The runtime pipeline (`avatar-core`) does **not** run diffusion; it only
interleaves the generated images according to the TTS phoneme output.

---

## 1. Inputs

| Input | Description |
|---|---|
| Reference image | One front-facing avatar photo/portrait (PNG/JPEG). Square preferred; will be center-cropped to square and downscaled. |
| Working resolution | **128×128 px** (matches the current `MAX_ATLAS_SOURCE`; can be raised later). |
| I2I engine | Candle SD-Turbo img2img (or any I2I diffusion that preserves identity well). |
| Identity conditioning | Required for consistency (ControlNet / IP-Adapter / LoRA / face-embedding). See §5. |

**Reference prompt** (base identity, used as the common prefix for all generations):

```
portrait photo of a [insert subject description], front-facing, neutral expression,
mouth closed, looking at camera, even lighting, sharp focus, photorealistic, 128x128
```

The agent must fill in `[insert subject description]` from the reference image
(e.g. "young woman with brown hair", "middle-aged man with glasses", "cartoon robot").
This prefix is prepended to every prompt below.

**Common negative prompt** (append to every generation):

```
blurry, low quality, distorted face, asymmetric eyes, extra limbs, deformed hands,
text, watermark, multiple faces, identity change, different person
```

**Common I2I settings** (for consistency):
- `denoising_strength`: **0.35–0.45** (low enough to preserve identity, high enough to change expression)
- `seed`: **fixed** (same seed for all images of one avatar)
- `steps`: SD-Turbo default (4); increase only if quality demands
- `guidance_scale`: SD-Turbo default (1.0); raise slightly for stronger prompt adherence

---

## 2. Output asset set

All outputs are **128×128 PNGs** (RGBA), named exactly as below, in a single
`avatar_assets/` directory. **Every image must show the same person/avatar.**

### 2.1 Expressions (7) — match `Mode` in `markers.rs`

| File | Prompt (append to reference prompt) |
|---|---|
| `expr_neutral.png` | `, neutral expression, relaxed face, mouth closed, looking at camera` |
| `expr_happy.png` | `, big genuine smile, eyes slightly squinting, joyful expression` |
| `expr_mad.png` | `, angry expression, furrowed brows, tense jaw, narrowed eyes` |
| `expr_sad.png` | `, sad expression, downturned mouth, drooping eyelids, somber look` |
| `expr_surprised.png` | `, surprised expression, raised eyebrows, wide eyes, mouth slightly open` |
| `expr_disgusted.png` | `, disgusted expression, wrinkled nose, raised upper lip, narrowed eyes` |
| `expr_fearful.png` | `, fearful expression, raised inner brows, wide eyes, tense mouth` |

### 2.2 Lip shapes / visemes (12) — match `VISEMES` in `visemes.rs`

The canonical viseme set is: `sil, AI, E, O, MBP, FV, L, TH, U, WQ, etc, rest`.

| File | Prompt (append to reference prompt) |
|---|---|
| `lip_sil.png` | `, mouth closed, lips gently together, neutral expression` |
| `lip_AI.png` | `, mouth wide open, jaw dropped, saying "ah" as in "father"` |
| `lip_E.png` | `, mouth moderately open, lips spread, saying "eh" as in "bed"` |
| `lip_O.png` | `, lips rounded and pushed forward, saying "oh" as in "go"` |
| `lip_MBP.png` | `, lips pressed firmly together, closed, as in "m", "b", "p"` |
| `lip_FV.png` | `, upper teeth lightly touching lower lip, as in "f" or "v"` |
| `lip_L.png` | `, tongue tip near upper front teeth, mouth slightly open, as in "l"` |
| `lip_TH.png` | `, tongue tip between teeth, mouth slightly open, as in "th"` |
| `lip_U.png` | `, lips in a small tight circle, saying "oo" as in "boot"` |
| `lip_WQ.png` | `, lips in a small rounded pucker, as in "w" or "wh"` |
| `lip_etc.png` | `, mouth slightly open, neutral mid-position, generic consonant` |
| `lip_rest.png` | `, relaxed mouth, neutral closed position, resting face` |

### 2.3 Head poses (5)

| File | Prompt (append to reference prompt) |
|---|---|
| `head_center.png` | `, head perfectly centered, facing camera directly` |
| `head_left.png` | `, head turned slightly to the left, still looking at camera` |
| `head_right.png` | `, head turned slightly to the right, still looking at camera` |
| `head_up.png` | `, head tilted slightly upward, chin raised, looking slightly up` |
| `head_down.png` | `, head tilted slightly downward, chin lowered, looking slightly down` |

### 2.4 Eyes / blinks (4)

| File | Prompt (append to reference prompt) |
|---|---|
| `eye_open.png` | `, eyes fully open, looking at camera, neutral expression` |
| `eye_blink.png` | `, eyes closed, eyelids shut, neutral expression` |
| `eye_squint.png` | `, eyes slightly squinting, relaxed lids, neutral expression` |
| `eye_wide.png` | `, eyes wide open, raised lids, alert expression` |

### 2.5 Total

**7 + 12 + 5 + 4 = 28 images** per avatar (all 128×128 PNGs).

---

## 3. Agent workflow

1. **Read the reference image.** Describe the subject (age, gender, hair, glasses,
   style, skin tone, etc.) → build the `[insert subject description]` prefix.
2. **Set up the I2I pipeline** (candle SD-Turbo img2img; see §5 for identity notes).
3. **Generate the reference baseline** (`expr_neutral.png`) first. Verify it matches
   the input identity closely. If not, adjust `denoising_strength` / conditioning.
4. **Generate all 28 images** with the fixed seed and common settings.
5. **Consistency check:** verify all images show the **same person** (same face shape,
   hair, skin, eyes). Reject/regenerate any that drift. This is the most important
   quality gate — see §5.
6. **Save** to `avatar_assets/` with the exact filenames above.
7. **Report:** list any images that needed regeneration, the final settings used
   (denoising strength, seed, steps, guidance), and the subject description used.

---

## 4. Runtime integration (how the assets are used)

The generated assets replace the current atlas-warp approach in `avatar-core`:

- **Expressions** (`expr_*.png`) → selected by the `<happy>`/`<mad>`/… markers,
  cross-faded over ~150 ms (existing `fade_t` mechanism).
- **Lip shapes** (`lip_*.png`) → selected by the phoneme→viseme mapping
  (`visemes.rs`, 12 visemes) from TTS phoneme timings.
- **Head poses** (`head_*.png`) → subtle procedural head bob/sway (existing
  `analysis.rs` head bob), or explicit pose changes.
- **Eyes** (`eye_*.png`) → procedural blinks (periodic) and expression-appropriate
  eye states.

Interleaving is a **lookup + cross-fade**, not diffusion — so runtime cost is
unchanged (fast, deterministic). The agent does **not** need to implement this;
it only produces the asset set.

---

## 5. Identity consistency (critical)

This is the single biggest risk. Vanilla img2img with different prompts will drift
between "different people". The agent **must** use identity-preserving conditioning.
Options, in order of preference:

1. **IP-Adapter / face-embedding conditioning** — inject the reference face embedding
   into the UNet. Best identity retention.
2. **ControlNet (landmark / expression conditioning)** — condition on face landmarks
   so expression changes are controlled while identity is pinned.
3. **LoRA fine-tuned on the reference avatar** — train a tiny LoRA (a few images) so
   the model knows this specific face.
4. **Fallback:** fixed seed + low denoising strength (0.35) + strong negative prompt.
   Acceptable for a prototype, but expect some drift.

> **Note:** candle does not currently ship ControlNet / IP-Adapter / LoRA training for
> SD. If robust identity is required, generate the assets **once in Python**
> (diffusers + ControlNet/IP-Adapter), export the 28 PNGs, and let Rust only do the
> runtime interleaving. The `avatar_assets/` format is engine-agnostic.

---

## 6. Acceptance criteria

- `avatar_assets/` contains exactly the **28 files** named in §2, all 128×128 PNG.
- All images show the **same person/avatar** (identity consistent).
- Each image clearly shows the requested expression / lip shape / pose / eye state.
- The neutral expression matches the reference image closely.
- A runtime smoke test (interleaving the assets with a TTS clip) produces a valid MP4
  with no missing-asset errors.

---

## 7. References

- `crates/avatar-core/src/markers.rs` — expression modes (`Mode` enum, 7 values).
- `crates/avatar-core/src/visemes.rs` — viseme set (`VISEMES`, 12 values) and
  phoneme→viseme mapping.
- `crates/avatar-core/src/tts.rs` — phoneme timings from espeak-ng/Piper.
- `crates/avatar-core/src/atlas.rs` — current atlas (to be replaced by assets).
- `crates/avatar-core/src/renderer.rs` / `renderer.wgsl` — runtime renderer
  (cross-fade, head bob, breathing).