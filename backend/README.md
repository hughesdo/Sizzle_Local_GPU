# `backend/` — the server

FastAPI app + the render pipeline. ~2,500 lines, no framework magic. Every
module carries a real docstring explaining *why* it does what it does — this
file is the map, not a substitute for reading them.

Nothing here is generated or vendored. **This directory is the irreplaceable
part of the backup**; everything else in the repo can be re-downloaded.

## Module map

| Module | Lines | Job |
|---|---:|---|
| `config.py` | 482 | **Start here.** Every tunable in one place, each with the reasoning behind its default. Also parses `.env`, resolves the CA bundle, and defines the model constraints (`snap_frames`, `validate_dimensions`). |
| `app.py` | 440 | FastAPI surface: audio/image upload, autoprompt, generate, the progress WebSocket, download, status badges. Serves `static/` as the SPA. |
| `jobs.py` | 435 | The single-GPU job queue. One worker thread, one FIFO. Owns the *timeline* semantics: image blocks, `continue` chaining, black gap fill, partial windows, and the graceful-failure rules. |
| `ltx_engine.py` | 589 | The GPU. Drives Lightricks' `A2VidPipelineTwoStage` in-process: stage 1 at half res → spatial upscale → stage 2 refine → decode. Applies the audio-reactive LoRA. |
| `audio.py` | 245 | librosa: waveform peaks for the UI, beat/downbeat detection, and proposed segment boundaries. |
| `mux.py` | 185 | ffmpeg final assembly. Concatenates clips, lays the **original** continuous audio over the top, NVENC with libx264 fallback. |
| `autoprompt.py` | 266 | The optional Anthropic vision call — the *only* outbound API in the app. Degrades to a heuristic stub with no key. |
| `apilog.py` | 83 | Admin-only JSONL ledger of Anthropic spend and GPU render times. Best-effort: never breaks a render. |

## The load-bearing design decisions

These are the things that will look like bugs if you don't know why:

**The generated audio is thrown away.** `ltx_engine.py` writes a *silent* mp4
even though the model generates audio. `mux.py` then lays the original,
continuous track over the concatenated timeline. That is deliberate: it means
the music never breaks at a clip seam. The generated audio only ever existed to
drive the reactive motion during sampling.

**Rendering is serialized behind one global lock.** One GPU, one worker thread,
one FIFO. This is not caution, it is load-bearing — see `MULTI-USER.md` §3 for
the full argument and what a second concurrent user actually experiences.

**A failed segment becomes black filler of the exact length, never a shorter
video.** Don's rule: never leave the user without their video. An image block
retries with a substitute image first (up to `RENDER_MAX_ATTEMPTS`); a
`continue` block falls back to black and the chain resumes on the next real
image. The mux stays gapless and the audio never desyncs.

**`config.py` reads `.env` with `setdefault`, so a real environment variable
always wins over the file.** That is how you inject secrets on a hosted box
without touching the committed `.env` (see `SETUP.md` §7). `SIZZLE_ENV_FILE`
repoints the file itself.

**All server state is in memory.** Jobs, clips and session ownership do not
survive a restart. `MULTI-USER.md` §4.1 has the consequences.

## Two model constraints that are not negotiable

Both live in `config.py` and everything routes through the helpers:

- **Frames must be `8n+1`** — the temporal VAE compresses time 8×. Use
  `snap_frames()`. Max 481 (20.04 s @ 24 fps); default cap 457.
- **Width and height must be multiples of 64** — stage 1 denoises at half the
  requested size and the upscaler doubles it, so `ltx` asserts on this. Use
  `validate_dimensions()`, which snaps rather than rejects.

Bypass either and you get an assertion from deep inside `ltx_core`, not a
helpful error.

## Rebuild notes

- **`ltx_engine.py` is the only module coupled to upstream.** It imports
  `ltx_pipelines` / `ltx_core` from the gitignored `LTX-2/` checkout. If a
  rebuild installs cleanly but fails at render time with an import or signature
  error, upstream drifted — check out the pinned commit in `VENDOR-PINS.txt`
  before debugging anything else.
- **Everything except `ltx_engine.py` runs without a GPU.** With
  `SIZZLE_BACKEND=mock`, `jobs.py` calls an ffmpeg synthesizer instead of the
  model, and the full timeline / continue / gap-fill / mux path is live. Use it
  to validate a rebuild before the 128 GB download finishes.
- **`config.missing_model_files()` is the single source of truth for
  readiness.** It is what the status badge, `check_weights.py` and the
  launchers all consult. If you add a required file, add it there.
- **No hardcoded GPU assumptions in the code** — VRAM strategy is entirely
  `SIZZLE_QUANTIZATION` / `SIZZLE_OFFLOAD` / resolution, all read from the
  environment. The *defaults* assume a ~96 GB card; the code does not. See
  `SETUP.md` §4.
- `ANTHROPIC_PRICING` in `config.py` is a hardcoded price table for the cost
  ledger. It will go stale; it affects reporting only, never behaviour.
