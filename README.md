# Sizzle — local GPU edition

**Turn a handful of still images into a beat-synced music video — entirely on
your own machine.** Drop in a track, drag images onto beat-aligned segments, hit
generate, and download a single muxed MP4 where every image "sizzles"
audio-reactively over its slice of the song.

This is [Sizzle](https://github.com/hughesdo/Sizzle) with the hosted rendering
pipeline torn out and replaced with **local inference on an NVIDIA GPU**. Same
UI, same timeline, same workflow — no fal.ai, no Cloudinary, no per-render bill.

> The original README called this "the interesting fork": *"if you've got a big
> GPU, swap the fal pipeline for local inference of the underlying model."*
> This is that fork.

> ### 📦 Rebuilding this on a different machine?
>
> **Read [`SETUP.md`](SETUP.md) instead of this file.** This README assumes a
> box that already works. `SETUP.md` is the from-scratch recovery procedure,
> written for the case where **the new GPU is not the one this was built on** —
> it covers the CUDA-wheel choice, the VRAM→settings map, hosted/rented GPU
> boxes, and a verification ladder.
>
> The defaults throughout this repo assume a **~96 GB card**. They are the
> highest-quality settings, not the safest ones. On anything smaller, read
> [`SETUP.md` §4](SETUP.md#4-tuning-for-a-gpu-that-isnt-a-96-gb-blackwell)
> before your first render.

---

## What changed

| | upstream Sizzle | this build |
|---|---|---|
| Rendering | fal.ai hosted endpoint | **LTX-2.3 in-process on your GPU** |
| Model | LTX-2.3 + fal audio-reactive LoRA | **the same** — locally |
| Media hosting | Cloudinary (or fal storage) | **none** — nothing is uploaded |
| Cost per 30s video | ~$2 of API credits | **$0** (electricity + your GPU) |
| Keys needed | `FAL_KEY` + Cloudinary creds | **none** |
| Anthropic auto-prompt | optional | **optional, unchanged** |
| Your images / music | uploaded to a third party | **never leave the machine** |

Everything else — the timeline, beat detection, continue-blocks, gap filling,
the graceful-substitution logic, the single-render lock, the Cloudflare tunnel —
works exactly as it did.

The one thing genuinely gone is the **content-policy pre-flight**. It existed
because fal would reject NSFW-leaning images mid-queue. Nothing rejects anything
now, so that check was re-pointed at something that still matters: whether a shot
is likely to get **age-gated or pulled by YouTube/TikTok/Instagram** once you post
the finished video. Same ⚠ badge, honest meaning.

## Fidelity to upstream Sizzle

Every generation parameter upstream sent to fal is reproduced locally, so a
timeline built on the hosted build renders the same way here:

| upstream sent to fal | this build | match |
|---|---|---|
| LoRA `ltx2.3_audio_reactive_lora.safetensors` | same file, 1,348,027,528 bytes, verified byte-for-byte | ✅ |
| `scale: 1.2` | `LORA_SCALE = 1.2` | ✅ |
| `transformer: "both"` | applied to stage 1 **and** stage 2 | ✅ |
| `image_strength: 0.62` | `ImageConditioningInput(frame_idx=0, strength=0.62)` | ✅ |
| `frames_per_second: 24` | `FPS = 24` | ✅ |
| `num_frames` (8n+1, authoritative) | same snapping, same authority | ✅ |
| `match_audio_length: False` | audio slice trimmed to exactly `num_frames/fps` | ✅ |
| `generate_audio: False` | clips written silent; original track muxed over | ✅ |
| `num_inference_steps: 8 / 20` | `8` distilled / `20` full | ✅ |
| `guidance_scale: 1` (both variants) | `1.0` distilled, **`3.0` full** | ⚠️ see below |

**The one deliberate deviation.** Upstream sent `guidance_scale: 1` for *both*
variants, because both hit the same hosted endpoint and the distilled/full toggle
only changed the step count. Running locally we have two *different* checkpoints,
and `ltx-2.3-22b-dev` is designed for real classifier-free guidance — at CFG 1 it
would render well below its ability. So "full quality" here means the dev
checkpoint at CFG 3.0, LTX-2.3's own default.

To get byte-for-byte upstream behaviour on the full variant instead:

```bash
SIZZLE_CFG_FULL=1.0                                  # match upstream's guidance
LTX_CKPT_FULL=models/ltx-2.3-22b-distilled.safetensors   # and its "same model, more steps"
```

**Image fitting had to be re-implemented.** Upstream did no image processing at
all — it handed fal a URL and fal fitted the picture on its servers, so the whole
image survived. Locally, ltx's conditioning path does a resize-and-*centre-crop*,
which silently ate the edges: a portrait still in a landscape frame kept only
**30.9%** of the picture. `SIZZLE_IMAGE_FIT=contain` (the default here) restores
the hosted behaviour; set `cover` for ltx's native crop.

**Resolution presets moved slightly** — 720 → 704, 1080 → 1088 — because the
two-stage pipeline requires multiples of 64 (see [Two hard model
constraints](#two-hard-model-constraints)). The hosted endpoint quietly fitted
odd sizes; running the model ourselves we have to be exact.

Everything else — beat detection, segmentation, continue-blocks, gap filling,
image substitution on failure, the mux — is upstream's code, untouched.

## Requirements

- **An NVIDIA GPU.** At the defaults (bf16, 22B model) budget **~48 GB of VRAM**.
  Smaller cards can still run it — see [Fitting a smaller card](#fitting-a-smaller-card).
- **~130 GB of free disk** for the model weights.
- **32 GB+ of system RAM** (64 GB+ recommended). Spare RAM is not wasted here:
  the OS page cache keeps the 43 GB checkpoint resident after the first read, so
  the second render stage loads it from memory rather than disk.
- **Windows or Linux**, Python 3.12, and **ffmpeg** on PATH (NVENC build ideally).

Developed and tested on an **RTX PRO 6000 Blackwell (96 GB)** with CUDA 13.2 and
PyTorch 2.13 `cu132` wheels (which cover Blackwell `sm_120` as well as older cards).

## Quickstart (Windows)

```
git clone <this repo>
cd Sizzle_Local_GPU
install.bat
```

`install.bat` installs `uv`, Python 3.12, CUDA PyTorch, ffmpeg, clones
Lightricks' LTX-2 inference packages, and offers to start the model download.
Then:

```
.venv\Scripts\python.exe scripts\download_models.py    # ~103 GB, resumable
startup.bat                                            # http://127.0.0.1:8000
```

- **`startup.bat`** — run it locally. Nothing is exposed to the internet.
- **`host.bat`** — share it with friends over a Cloudflare quick tunnel.
- **`GPUQuery.bat`** — what GPU is *really* in this machine (see below).

macOS/Linux use `./run.sh` and `./host.sh` — the same two jobs, non-Windows
flavour. CUDA is still required for real rendering.

### Don't trust Windows about your GPU

`GPUQuery.bat` exists because Windows misreports big cards.
`Win32_VideoController.AdapterRAM` is a **32-bit field**, so anything with 4 GiB
or more wraps around and reports ~4.29 GB — and Device Manager, dxdiag and Task
Manager all read that field. On the development box, 96 GB of Blackwell shows up
in Windows as "4.00 GB", listed underneath a "Microsoft Hyper-V Video" adapter
because the card is passed through into a VM.

`GPUQuery.bat` asks the driver and the CUDA runtime instead, prints Windows'
answer alongside so the discrepancy is obvious rather than alarming, confirms
PyTorch has native kernels for your compute capability (rather than JIT-ing from
PTX on every start), and ends with a VRAM verdict — whether to run bf16, fp8 or
NVFP4 at your card's size.

### While the weights download

Set `SIZZLE_BACKEND=mock` and the whole app works with no model at all — ffmpeg
synthesizes each clip from your still, so you can build timelines, watch the
queue, and get a real muxed MP4 out. It is the best way to learn the pipeline,
and it exercises every path except the model itself.

## The Gemma gate (the one manual step)

LTX-2.3 encodes prompts with **Gemma 3 12B**, and Google gates that repo. Once:

1. Accept the license at
   [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
2. `.venv\Scripts\hf.exe auth login` (or set `HF_TOKEN=hf_...`)
3. `python scripts/download_models.py --only gemma`

Everything else downloads with no account at all.

> Lightricks also ship an **ungated** copy of the same Gemma 3 12B inside
> [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2) (`text_encoder/` +
> `tokenizer/`). It is stored fp32, so it is ~49 GB instead of ~24 GB, and it is
> the copy paired with their 19B model rather than the QAT variant LTX-2.3's docs
> name. If you'd rather not accept the Gemma license, assemble that into
> `models/gemma/` and point `GEMMA_PATH` at it.

## How it works

Each image is a **first frame** that the audio-reactive LoRA animates for the
length of its segment. Beats are detected with `librosa`, segments snap to beats,
and the final assembly overlays the **original continuous audio** so the music
never breaks at a seam.

Per clip, locally:

```
 stage 1   denoise video at HALF resolution, audio latent FROZEN as conditioning
 upsample  2x the video latent (spatial upscaler)
 stage 2   refine at full resolution with the distilled LoRA
 decode    VAE -> RGB frames -> silent mp4
 mux       concat all clips, lay the original track over the top
```

The generated audio is thrown away on purpose — the model's job here is motion,
and the music you hear is always your original file, unbroken.

```
sizzle/
  backend/
    app.py         FastAPI: endpoints + websocket + static serving
    config.py      all paths, params, and env-var wiring (+ .env loading)
    audio.py       librosa beat detection, segmentation, slicing
    ltx_engine.py  render backends: local (GPU) | mock (ffmpeg)
    jobs.py        single-lane FIFO job queue, per-segment render loop
    mux.py         ffmpeg concat + continuous-audio overlay
    autoprompt.py  optional Anthropic vision prompt + NSFW pre-flight
    apilog.py      per-call ledger (Anthropic tokens, local render timings)
  scripts/
    download_models.py  fetch every weight this app needs
    check_weights.py    is this box ready to render? (drives the launchers)
    gpu_report.py       the real GPU, vs what Windows claims
    test_local.py       smoke-test the GPU render path end to end
    test_pipeline.py    exercise the whole job loop on the mock backend
    test_api.py         drive a running server over HTTP + websocket
  static/          the single-page timeline UI
  models/          the weights (gitignored)
  LTX-2/           Lightricks' inference packages (cloned by install.bat)

  install.bat      one-time setup          GPUQuery.bat  GPU diagnostics
  startup.bat      run locally             host.bat      share over a tunnel
  run.sh           run locally (mac/linux) host.sh       share (mac/linux)
```

## Two hard model constraints

Both are enforced for you, but they explain why some numbers look odd:

- **Frame counts must be `8n+1`** (the temporal VAE compresses time 8×). Every
  count goes through `config.snap_frames()`. Clips cap at 457 frames = 19.04s.
- **Width and height must be multiples of 64.** The two-stage pipeline renders at
  half size then upscales 2×, so it asserts this. The hosted endpoint was
  permissive and quietly fitted odd sizes; running the model ourselves we have to
  be exact. This is why the YouTube preset is **1280×704**, not 1280×720, and the
  HD ones are 1088 rather than 1080. Custom sizes snap to the grid as you type.

## Measured performance

On the RTX PRO 6000 Blackwell (96 GB), distilled variant, bf16, 768×448, a 3.04s
clip (73 frames):

| | |
|---|---|
| first clip of a session (cold, loads ~66 GB) | **77s** |
| every clip after that (warm) | **~67s** (22× realtime) |
| peak VRAM | **39.6 GB** of 95.6 |
| a 60s video (~20 clips) | **~22 min** of GPU time |

**Most of that is not compute.** Profiling one warm clip: ~15s Gemma load +
prompt encode, ~19s building the stage-1 transformer, ~4s actually denoising,
~23s building the stage-2 transformer, ~3s stage-2 denoise + VAE decode. The GPU
averages **16.6% utilisation and 86 W of its 600 W** across a render, and is
≥50% busy only 7.8% of the time — the bottleneck is reading and LoRA-fusing the
43 GB checkpoint twice per clip, not the maths.

That leaves a lot on the table. See the last section of
[`MULTI-USER.md`](MULTI-USER.md) for the concrete ways to claw it back (batching
stages per job rather than per clip is the big one) and for why simply rendering
two clips concurrently does **not** work today.

## Tuning

Everything below goes in `.env` (which is now actually read — see
[`.env.example`](.env.example) for the full list).

**Speed vs quality**

| knob | default | effect |
|---|---|---|
| variant dropdown | `distilled` | 8 steps, CFG 1. The `full` (dev) checkpoint runs 20 steps with real guidance — slower, better |
| `SIZZLE_QUANTIZATION` | `none` (bf16) | `fp8-scaled-mm` roughly halves weight VRAM **and is faster on Blackwell**; `nvfp4-cast` is smaller/faster still, lowest fidelity |
| output format | 1280×704 | render time scales with pixels — the dropdown shows MP/frame |
| `SIZZLE_CACHE_WEIGHTS` | `0` | caches checkpoint tensors between the two stages. Leave it off: ltx_core keeps that cache on the **build device**, so on CUDA it pins ~43 GB in VRAM *on top of* the model built from it. The OS page cache already keeps the file in spare RAM for free |
| `SIZZLE_WARMUP` | off | build the pipeline at server start instead of on the first Generate |

**Look**

| knob | default | effect |
|---|---|---|
| `SIZZLE_LORA_SCALE` | `1.2` | audio-reactive LoRA strength (the LoRA card suggests 1.2–1.5) |
| `SIZZLE_LORA_V2` | off | the v2 LoRA: stronger beat impact, heavier motion |
| `SIZZLE_IMAGE_STRENGTH` | `0.62` | how tightly the clip is pinned to your still |
| `SIZZLE_IMAGE_FIT` | `contain` | `contain` keeps the whole image and letterboxes; `cover` fills the frame and crops the edges off (ltx's native behaviour); `stretch` distorts |
| `SIZZLE_A2V_GUIDANCE` | `3.0` | how hard the music steers the motion |

Prompting matters a lot with this LoRA. Lead with something like
`sound-driven video, audio-reactive motion, continuous visual flow` and be
explicit about what hits the beat — the auto-prompt does this for you.

### Fitting a smaller card

- `SIZZLE_QUANTIZATION=fp8-scaled-mm` — ~24 GB of weights, faster on Blackwell.
- `SIZZLE_QUANTIZATION=nvfp4-cast` — smaller again on Blackwell.
- `SIZZLE_OFFLOAD=cpu` — stream layers from pinned RAM (~5 GB VRAM, much slower).
- `SIZZLE_OFFLOAD=disk` — lowest memory of all, slowest.
- Drop the output format and shorten blocks; attention cost grows with both.

## Verifying it works

```bash
python scripts/test_local.py                  # real GPU render, synthetic inputs
python scripts/test_local.py --variant full --seconds 5
SIZZLE_BACKEND=mock python scripts/test_pipeline.py   # whole job loop, no GPU
```

`test_local.py` reports wall-clock, realtime factor and peak VRAM, then ffprobes
the result so a clip with the wrong frame count fails loudly.

## The Anthropic key (optional but recommended)

`ANTHROPIC_API_KEY` is the **only** outbound call this build can make. When set, a
Claude vision model looks at each image and (1) writes a tailored audio-reactive
prompt for that specific frame — far better than the generic fallback — and
(2) rates how likely the shot is to be age-gated by a social platform. Without
it, both are skipped and every segment uses the same generic reactive prompt.
Set it in `.env`.

## Serving it to friends (Cloudflare tunnel)

`host.bat` (or `./host.sh`) starts the server and prints a
`https://something.trycloudflare.com` URL. Hand that to a friend. Requires
[`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
on PATH (`winget install --id Cloudflare.cloudflared`).

Renders now run on **your GPU** rather than your API credits — one at a time,
because the app holds a global single-render lock. The random URL is your only
access control: share it with people you trust and close the window when done.

**Sharing it with more than one person?** Read [`MULTI-USER.md`](MULTI-USER.md).
Short version: uploads, timelines, clips and finished videos are all properly
separated per browser session — nothing gets tangled. But rendering is strictly
one-at-a-time and the second person is *rejected rather than queued*: their
GENERATE button greys out to "someone is rendering…" and they have to click
again once the GPU frees up.

## Known gaps

- No transitions between clips yet (hard cuts).
- One resolution per job.
- Switching the variant dropdown mid-session reloads a 46 GB checkpoint.
- First render of a session pays the model-load cost unless `SIZZLE_WARMUP=1`.

## Documentation map

| Document | Covers |
|---|---|
| **[`SETUP.md`](SETUP.md)** | **Rebuilding from scratch on another GPU box.** Start here on a new machine. |
| [`models/README.md`](models/README.md) | The weights: full manifest, download, the Gemma gate, relocating them |
| [`MULTI-USER.md`](MULTI-USER.md) | What happens when you share the tunnel URL — isolation and the render lock |
| [`backend/README.md`](backend/README.md) | Module-by-module tour of the server and its load-bearing decisions |
| [`scripts/README.md`](scripts/README.md) | Every script, what it proves, and the order to run them in |
| [`static/README.md`](static/README.md) | The front-end and its contract with the API |
| [`certs/README.md`](certs/README.md) | The TLS-inspection workaround |
| [`_work/README.md`](_work/README.md) | Runtime scratch — what's in it, and moving it to persistent storage |
| [`VENDOR-PINS.txt`](VENDOR-PINS.txt) | Exact upstream commits this build was validated against |
| [`.env.example`](.env.example) | Every configuration knob, annotated — the source of truth |
| [`requirements.lock.txt`](requirements.lock.txt) | Exact package versions from the working box |

## Licenses

The app is [CC BY 4.0](LICENSE) © 2026 Don Hughes — use it, fork it, improve it,
sell it, just give credit. The **model weights are not mine to license**: LTX-2.3,
the spatial upscaler, the distilled LoRA and the fal audio-reactive LoRA are all
covered by the [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md),
and Gemma 3 by Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
Read them before shipping anything commercial.
