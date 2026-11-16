# PLAN_REMOVE_POSE_WARP.md — Remove step [5] (pose warp) from the talking-head pipeline

**Status: implemented (Tasks 1–10; all committed).**

## Goals

1. Remove **step [5] (pose warp)** from the talking-head render pipeline. New
   pipeline:

   ```
   src ─▶ [0] square N×N ─▶ [1] bbox crop M×M → 96×96 ─▶ [2] Wav2Lip(clean)
       ─▶ [3] 96×96 → 112×112, expression warp (PFLD) ─▶ [4] 112×112 → M×M, in-paint into N×N
       ─▶ [5] resize ─▶ frame
   ```

2. Remove **all** machinery that existed only to feed step [5]: the Audio2Pose
   decode, the Wav2Lip `audio_embeddings` pass and method, the `pose` crate, the
   `pose_seed` API knob, the `pose.bpk` weights, and the Audio2Pose export.
3. Keep the **expression warp (step 3)** and the **in-paint (step 4)** intact —
   `strength` still drives the expression warp and **stays**.
4. Update docs so the documented pipeline matches the code.

**Expected saving** (agreed estimate, not re-measured): ~0.4–0.8 s on a 3 s /
75-frame CPU render (~3–5 % of 14.3 s), **plus** the removed
`wav2lip.audio_embeddings` pass (a full Wav2Lip audio-encoder forward over the
clip) and the Audio2Pose decode.

## Confirmed decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Scope | **Full removal** (Tasks 1–10) |
| 2 | `pose_seed` API | **Remove** (breaking; no external consumers) |
| 3 | `Wav2Lip::audio_embeddings` | **Delete** the method + `Model::audio_embedding` + test |
| 4 | `crates/pose` | **Delete** the crate |
| 5 | `pose.bpk` | **Delete** the file + its `CHECKSUMS.txt` entry |
| 6 | `strength` | **Keep** (drives the expression warp) |
| 7 | `convert_weights.py` | **Drop** the Audio2Pose export |
| 8 | External consumers | **None** — safe to break the API |

## Current behavior

`render_talking_head` (`crates/avatar-core/src/talking_head.rs`) runs 8 stages
(doc header, lines 1–21):

1. resample → 16 kHz
2. ExpNet → per-frame 64-d expression coefficients
3. **5b** — `wav2lip.audio_embeddings(&chunks, 8)` → `pose::Audio2Pose::decode`
   → `pose_seq` (6 values per frame)
4. step 0 — square N×N
5. step 1 — bbox crop M×M → 96×96
6. step 2 — Wav2Lip per mel chunk
7. per frame: step 3 (PFLD + expression warp, 112×112), step 4 (in-paint into
   N×N), **step 5 (`fusion::pose_warp` on the whole N×N frame)**, step 6 (resize)
8. encode/mux

## Expected behavior

- Frames identical to today **minus** the global affine head motion; the mouth
  (Wav2Lip) and expression warp are unchanged.
- The render is **fully deterministic** (no stochastic latent) — repeated runs
  with identical inputs are byte-identical.
- No `pose.bpk` required; no `pose_seed` parameter; no `pose` crate.

## Verification with unit tests

| Test | Action |
|---|---|
| `talking_head_renders_short_clip` | keep |
| `talking_head_audio_drives_frames` | keep |
| `talking_head_pose_seed_determinism` | **delete** |
| **new** `talking_head_render_is_deterministic` | two renders, identical bytes |
| `fusion::pose_changes_frame` | **delete** |
| `fusion::small_realistic_pose_moves_frame_visibly` | **delete** |
| `fusion::pose_warp_is_bounded` | **delete** |
| `wav2lip::audio_embeddings_shape_and_vary` | **delete** |
| `fusion::landmark_warp_*` (expression warp) | keep — untouched |

Plus: `cargo test --workspace`, `cargo clippy --workspace`, and a manual
end-to-end render (`python examples/render_clip.py`).

## Tasks (one commit per task; confirm before starting the next)

> Each task is small, self-contained, and leaves the tree compiling + green.
> Task 2 removes the `pose_seed` parameter from `render_talking_head` and
> updates **all** call sites (tests, bench, `pipeline.rs`) in the same commit so
> the tree keeps building; Task 6 then removes the `EngineConfig`/Python/CLI knob.

### Task 1 — Drop step 5 from the render loop
- `talking_head.rs`: delete the step-5 block (lines 242–256); pass `square2`
  straight to step 6; renumber the step-6 comment to step 5.
- Update the per-frame comment block (lines 209–214) and the module doc
  (lines 1–21) to the new 5-step order.
- **Commit:** `talking_head: remove step 5 (pose warp) from the render loop` —
  body: what was removed (the `pose_warp` call + `pose_row` lookup), that
  `square2` now feeds the resize directly, and the new step order.
- **Verify:** `cargo test -p avatar-core`.

### Task 2 — Remove the Audio2Pose wiring from `render_talking_head`
- Delete `POSE_WEIGHTS` (line 46), the pose weights load/`ensure!` (132, 145–147),
  the `Audio2Pose` construction (160–162), the `audio_embeddings` + `decode`
  block (174–182), and the `pose_seed` parameter (111 + doc 97–99).
- Update **all** call sites to drop the `pose_seed` argument: `talking_head.rs`
  tests (404, 431–432, 476, 528, 592, 642–644, 657), `talking_head_bench.rs`
  (28), `pipeline.rs` (262).
- Delete the `talking_head_pose_seed_determinism` test; add
  `talking_head_render_is_deterministic`; drop the `pose` entry from
  `models_present()` (371).
- **Commit:** `talking_head: remove Audio2Pose wiring and pose_seed parameter` —
  body: lists each removed item (weights, model, embeddings pass, decode,
  parameter, test) and the new determinism test.
- **Verify:** `cargo test -p avatar-core`.

### Task 3 — Remove `fusion::pose_warp`
- Delete `pose_warp` (lines 209–279) and the three pose tests (452–517); update
  the `fusion.rs` module doc (lines 1–21) to drop the pose-warp bullet and the
  "pose warp" mention in the orchestration sentence.
- **Commit:** `fusion: remove pose_warp and its tests` — body: notes the
  expression warp (`warp_from_landmarks`) is unaffected and is now the only warp.
- **Verify:** `cargo test -p avatar-core`.

### Task 4 — Remove the Wav2Lip audio-encoder embedding API
- `crates/wav2lip/src/lib.rs`: delete `audio_embeddings` (~70–105),
  `AUDIO_EMB_DIM` (35), the test `audio_embeddings_shape_and_vary` (461–485),
  and the "Audio2Pose — Plan 6" doc mention.
- `crates/wav2lip/src/model/wav2lip.rs`: delete `audio_embedding` (~528–573)
  and its Audio2Pose doc mention.
- **Commit:** `wav2lip: remove the audio-encoder embedding API (Audio2Pose only)`
  — body: notes the method was used solely by Audio2Pose and that `forward`
  (the full model) is unchanged.
- **Verify:** `cargo test -p wav2lip`.

### Task 5 — Remove the `pose` crate from the workspace
- Root `Cargo.toml`: drop `"crates/pose"` from `members`.
- `avatar-core/Cargo.toml`: drop `pose = { path = "../pose" }` and its comment.
- Delete `crates/pose/` (Cargo.toml, src/lib.rs, src/model/{mod,pose}.rs).
- **Commit:** `workspace: remove the pose (Audio2Pose) crate` — body: lists the
  removed member, dependency, and crate files.
- **Verify:** `cargo test --workspace`.

### Task 6 — Remove `pose_seed` from the public API
- `pipeline.rs`: drop the `pose_seed` field (35–38), its default (87),
  `with_pose_seed` (130–133), and the argument in the `render_talking_head` call
  (262 — already dropped in Task 2; remove the now-stale comment).
- `avatar-py/src/lib.rs`: drop the field (32), the `#[pyo3(signature = ...)]`
  entry (60), the constructor parameter (69), the doc line (53), the struct
  init (110), the field copy (131), and the `pose_seed()` getter (149–152).
- `examples/render_clip.py`: drop `--pose-seed` (16, 164) and the `pose_seed=`
  kwarg (180).
- **Commit:** `api: remove pose_seed from EngineConfig, Python bindings and CLI`
  — body: notes this is a breaking change for callers passing `pose_seed`.
- **Verify:** `cargo test --workspace`; `cargo build -p avatar-py`.

### Task 7 — Delete the pose weights
- Delete `.res/models/pose/` (`pose.bpk`, `pose.onnx.txt`, `pose.rs`).
- `.res/models/CHECKSUMS.txt`: remove the `models\pose\pose.bpk` and
  `pose.onnx` lines.
- **Commit:** `res: delete the pose weights and their checksums` — body: lists
  the deleted files and checksum lines.
- **Verify:** `cargo test --workspace` (the talking-head tests skip when weights
  are absent).

### Task 8 — Drop the Audio2Pose export from the scripts
- `scripts/convert_weights.py`: delete `PoseResidualConv` / `PoseUpsample` /
  `PoseResUnet` / `Audio2PoseDecoder` / `load_pose_state` / `export_pose`, the
  `--pose` / `--skip-pose` args, the `pose_ckpt` default, the export block, and
  the Audio2Pose mentions in the module docstring.
- `scripts/verify_onnx.py`: delete `verify_pose()` and its imports/call.
- `scripts/README_WEIGHTS.md`: remove the Audio2Pose rows/sections.
- **Commit:** `scripts: drop the Audio2Pose weight export` — body: lists the
  removed classes/functions/args in each script.
- **Verify:** `python -c "import ast; ast.parse(open('scripts/convert_weights.py').read())"`
  (syntax check; no torch run).

### Task 9 — Update the documentation
- `steps-analysis.md`: pipeline diagram (drop step 5, renumber step 6 → 5), the
  per-step sections, the data-flow table, the "Current vs new" block, and the
  decision/notes rows mentioning the pose warp.
- `README.md`: ASCII pipeline (34–35), crate table (44–45), `pose_seed` docs
  (24, 157, 176, 183–184), model list (85), test count (54), changelog (223),
  and the intro (7, 10, 73).
- `PERF_NOTES.md`: drop the pose warp from the stage list (27), the
  `audio_embeddings` mention (41), and the "where the time goes" / "cap N"
  mentions (60, 69).
- **Commit:** `docs: update pipeline docs after removing the pose warp` — body:
  lists the three files and the sections updated.
- **Verify:** read-back / diff review (no code change).

### Task 10 — Final end-to-end verification
- `cargo test --workspace`, `cargo clippy --workspace`.
- Render a clip (`python examples/render_clip.py`) and confirm the MP4 plays and
  the frame count/size are correct.
- **Commit:** only if a fix is needed; otherwise report results and skip.
- **Verify:** the rendered MP4 + the green test suite.

## Files touched (summary)

| File | Tasks |
|---|---|
| `crates/avatar-core/src/talking_head.rs` | 1, 2 |
| `crates/avatar-core/src/fusion.rs` | 3 |
| `crates/wav2lip/src/lib.rs`, `.../model/wav2lip.rs` | 4 |
| root `Cargo.toml`, `crates/avatar-core/Cargo.toml`, `crates/pose/**` | 5 |
| `crates/avatar-core/src/pipeline.rs`, `crates/avatar-py/src/lib.rs`, `examples/render_clip.py` | 6 |
| `.res/models/pose/**`, `.res/models/CHECKSUMS.txt` | 7 |
| `scripts/convert_weights.py`, `scripts/verify_onnx.py`, `scripts/README_WEIGHTS.md` | 8 |
| `steps-analysis.md`, `README.md`, `PERF_NOTES.md` | 9 |

## Remaining note (not blocking)

- `.res/` also holds the **raw** SadTalker sources used only to build
  `pose.onnx`: `auido2pose_00140-model.pth`, `pose.onnx`, `pose.onnx.data`,
  `sadtalker_cvae.py`, `sadtalker_res_unet.py`, `sadtalker_auido2pose.yaml`.
  Task 7 deletes only the runtime weights (`.res/models/pose/`) + checksums.
  **Tell me if you also want these raw files removed** (they are ~91 MB + the
  ONNX); otherwise they stay.

## Implementation notes (deviations from the plan)

- **Task 1** also swapped the `talking_head_pose_seed_determinism` test for
  `talking_head_render_is_deterministic` (that test asserted the removed step-5
  stochastic behaviour and would otherwise have failed Task 1's verification).
- **Task 7** produced **no commit**: the whole `.res/` tree is git-ignored
  (`.gitignore` line 7, `.*/`), so the weight deletion + `CHECKSUMS.txt` edit are
  filesystem-only. Left as-is per user.
- **Task 8** also covered `scripts/convert_weights.ps1` (not in the plan's file
  list), which drove the same `pose.onnx` → `pose.bpk` conversion.
- **Task 10** produced **no commit** (verification only). `cargo test
  --workspace` passes; `cargo clippy --workspace` still fails on two
  **pre-existing** `clippy::erasing_op` errors in `wav2lip::face_input`
  (`lib.rs:253`), unrelated to this plan (fixed separately).
