# SETUP — rebuilding Sizzle on a different GPU box

`README.md` explains *what this project is* and how to use it on a machine
that already works. **This file assumes nothing exists yet.** It is the
recovery procedure: bare host → rendering video.

It is written for the case that actually matters here — **the new GPU is not
the one this was built on.** Everything hardware-specific is called out with
what to change and why.

**Contents**

1. [What you are rebuilding](#1-what-you-are-rebuilding)
2. [Hardware requirements, and what's negotiable](#2-hardware-requirements-and-whats-negotiable)
3. [The rebuild, step by step](#3-the-rebuild-step-by-step)
4. [Tuning for a GPU that isn't a 96 GB Blackwell](#4-tuning-for-a-gpu-that-isnt-a-96-gb-blackwell)
5. [Hosted / rented GPU boxes](#5-hosted--rented-gpu-boxes)
6. [What is *not* in this repo, and why](#6-what-is-not-in-this-repo-and-why)
7. [The `.env` in this repo — read this](#7-the-env-in-this-repo--read-this)
8. [Verification ladder](#8-verification-ladder)
9. [Troubleshooting the rebuild](#9-troubleshooting-the-rebuild)

---

## 1. What you are rebuilding

Sizzle turns a still image + a music track into a beat-synced video. This
edition renders **entirely on a local NVIDIA GPU** — Lightricks' LTX-2.3 22B
model plus fal's audio-reactive LoRA, driven in-process. There is no fal.ai
account, no Cloudinary, no media hosting.

Four moving parts, and you need all four:

| Part | Where it comes from | Size | In git? |
|---|---|---:|:---:|
| **This repo** — FastAPI backend, browser UI, scripts | you're looking at it | ~1 MB | ✅ |
| **Python env** — CUDA PyTorch + deps | `install.bat` / `install.sh` | ~8 GB | ❌ rebuilt |
| **LTX-2 packages** — the inference engine | cloned from Lightricks | ~8 MB | ❌ cloned |
| **Model weights** — checkpoints, LoRAs, Gemma | Hugging Face | ~123 GB | ❌ downloaded |

The only irreplaceable one is the first. The other three are recipes, and this
document is those recipes.

**The single outbound API call this app can make** is the optional Anthropic
vision auto-prompt. Without an `ANTHROPIC_API_KEY` it falls back to a generic
prompt and the app is 100 % offline.

---

## 2. Hardware requirements, and what's negotiable

### Hard requirements

- **An NVIDIA GPU with CUDA.** Not negotiable. AMD/Apple/CPU will not render.
  (`SIZZLE_BACKEND=mock` runs the whole app *around* the model on any machine,
  which is genuinely useful for UI work — but it produces ffmpeg slideshows,
  not model output.)
- **~150 GB of free disk** — 123 GB resident plus download staging headroom.
- **Python 3.12.** `uv` installs it for you; you do not need it preinstalled.
- **ffmpeg on `PATH`**, ideally an NVENC-enabled build.

### VRAM — the number that decides your config

| VRAM | Verdict | What to set |
|---:|---|---|
| **80 GB+** | Everything at defaults. | nothing — defaults assume this |
| **48–80 GB** | bf16 fits. This is the real floor for the default config. | nothing, but watch for OOM at 1920×1088 |
| **24–48 GB** | Needs quantization. | `SIZZLE_QUANTIZATION=fp8-scaled-mm` |
| **16–24 GB** | Tight. Quantize *and* offload, expect it to be slow. | `fp8-scaled-mm` or `nvfp4-cast` + `SIZZLE_OFFLOAD=cpu`, smaller resolutions |
| **< 16 GB** | Realistically, no. | `SIZZLE_BACKEND=mock` |

Details and the reasoning behind each in [§4](#4-tuning-for-a-gpu-that-isnt-a-96-gb-blackwell).

### System RAM

**32 GB minimum, 64 GB+ strongly recommended** — and spare RAM is not wasted
here. The two render stages each read the 46 GB checkpoint; the OS page cache
keeps it resident between them, so stage 2 loads from memory instead of disk.
That is the "cache the weights in RAM" mechanism that actually works, and it
needs no configuration. (See the long comment on `CACHE_WEIGHTS_IN_RAM` in
`backend/config.py` for why the *explicit* caching flag is off by default and
should stay off — it caches into **VRAM**, not RAM, and makes things worse.)

### The reference box (what "known good" means)

Everything in this repo was developed and measured on:

```
GPU      NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB (96 GB)
         compute capability 12.0  (sm_120)
Driver   595.97          CUDA 13.2
Torch    2.13.0+cu132    torchaudio 2.11.0+cu132
Python   3.12.13
OS       Windows 10 Pro 19045 (the card is passed through into a VM)
ffmpeg   8.1.2-full_build (gyan.dev), NVENC enabled
```

**Assume none of that is true on the new box.** The parts that matter and why
are in [§4](#4-tuning-for-a-gpu-that-isnt-a-96-gb-blackwell).

---

## 3. The rebuild, step by step

### 3.0 — Get the code

```bash
git clone https://github.com/<you>/Sizzle_Local_GPU.git
cd Sizzle_Local_GPU
```

### 3.1 — Run the installer

**Windows:**
```bat
install.bat
```

**Linux / macOS:**
```bash
chmod +x install.sh && ./install.sh
```

Both do the same eight things, and both are **safe to re-run** — they reuse an
existing `.venv` and never overwrite an existing `.env`:

1. Install `uv` if missing (winget on Windows, the official installer script
   elsewhere).
2. Install `ffmpeg` if missing (winget / apt / dnf / brew, best effort).
3. Sanity-check `nvidia-smi` and print what the driver reports.
4. **Clone `Lightricks/LTX-2`** into `./LTX-2` (gitignored — see
   `VENDOR-PINS.txt` for the exact validated commit).
5. Install Python 3.12 via `uv` and create `.venv`.
6. **Install CUDA PyTorch from PyTorch's own index** — this is the step that
   most often needs adjusting on a different GPU, see below.
7. `pip install -e` the two LTX packages, then `-r requirements.txt`.
8. Copy `.env.example` → `.env` if `.env` is absent. *(In this backup `.env` is
   already committed, so this is a no-op — which is intended.)*

Then it verifies `torch.cuda.is_available()` and that the LTX pipeline class
imports, and offers to start the model download.

> ### ⚠️ Step 6 is the GPU-specific one
>
> The installers pin **`torch==2.13.0+cu132`** from
> `https://download.pytorch.org/whl/cu132`. PyPI's default `torch` is
> **CPU-only** and would silently give you a machine that cannot render — hence
> the explicit index.
>
> `cu132` was chosen because Blackwell (`sm_120`) requires it, and it also
> covers every older CUDA card. **On most hosts you should leave it alone.**
>
> Change it only if the new host's driver is too old for CUDA 13.2. Both
> installers honour an override so you don't have to edit them:
>
> ```bash
> SIZZLE_TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
> SIZZLE_TORCH_SPEC=torch \
> ./install.sh
> ```
> ```bat
> set SIZZLE_TORCH_INDEX=https://download.pytorch.org/whl/cu128
> set SIZZLE_TORCH_SPEC=torch
> install.bat
> ```
>
> Rule of thumb — match the CUDA build to what `nvidia-smi` prints in its
> top-right "CUDA Version" field (that is the **maximum** the driver supports;
> a wheel built for that or lower will work):
>
> | Driver reports | Use index | Notes |
> |---|---|---|
> | 13.x | `.../whl/cu132` | required for Blackwell sm_120 / RTX 50xx / RTX PRO |
> | 12.8–12.9 | `.../whl/cu128` | fine for Hopper, Ada, Ampere |
> | 12.6 | `.../whl/cu126` | older Ampere/Turing hosts |
>
> If you get it wrong the symptom is loud and immediate: either
> `torch.cuda.is_available()` is `False`, or you get
> `no kernel image is available for execution on the device` on the first
> render. Both mean "wrong wheel", not "broken GPU".

### 3.2 — Download the weights

```bash
# Windows
.venv\Scripts\python.exe scripts\download_models.py
# Linux / macOS
./.venv/bin/python scripts/download_models.py
```

~128 GB, resumable, idempotent. **One step is manual: Gemma 3 12B is a gated
Hugging Face repo** and you must accept its licence once, then authenticate.

👉 **Full instructions, the file manifest, the fastest partial-download order,
and the ungated Gemma alternative are in [`models/README.md`](models/README.md).**

### 3.3 — Configure

`.env` is committed with this backup and is already set to the tuned defaults
(`SIZZLE_BACKEND=local`, everything else commented at its documented default).
**On a card smaller than ~48 GB you must edit it** — see
[§4](#4-tuning-for-a-gpu-that-isnt-a-96-gb-blackwell).

`.env.example` is the fully-annotated reference for every knob and is kept
current; read it rather than guessing.

### 3.4 — Run

```bat
startup.bat        :: Windows, http://127.0.0.1:8000
host.bat           :: Windows, + Cloudflare quick tunnel for sharing
```
```bash
./run.sh           # Linux/macOS, http://127.0.0.1:8000
./host.sh          # Linux/macOS, + Cloudflare quick tunnel
```

Both launchers run `scripts/check_weights.py` first and **warn without
blocking** — the timeline editor, waveform and auto-prompt all work with no
weights at all; only **Generate** needs them.

Sharing the tunnel URL? Read [`MULTI-USER.md`](MULTI-USER.md) first — rendering
is deliberately serialized behind one global lock, and that document explains
exactly what is and isn't isolated between users.

---

## 4. Tuning for a GPU that isn't a 96 GB Blackwell

**The defaults in this repo assume ~96 GB of VRAM.** They are the *highest
quality* settings, not the *safest* ones. This section is the honest map of
what to change.

Start by asking the driver, not the OS:

```bash
python scripts/gpu_report.py     # VRAM, compute capability, kernel coverage, a verdict
```
```bat
GPUQuery.bat                     :: Windows wrapper for the same thing
GPUWatch.bat                     :: live VRAM/util while a render runs
```

> **Do not trust Windows about VRAM.** `Win32_VideoController.AdapterRAM` is a
> **32-bit** field, so anything ≥ 4 GiB wraps and reports ~4.29 GB — and Device
> Manager, dxdiag and Task Manager all read that field. On the reference box,
> 96 GB of Blackwell shows up in Windows as "4.00 GB". `GPUQuery.bat` asks the
> driver and the CUDA runtime instead and prints Windows' wrong answer beside
> the right one so the discrepancy looks like a known quirk rather than a
> failure.

### The three knobs that matter, in the order to reach for them

**1. Quantization** — `SIZZLE_QUANTIZATION` (default `none` = bf16)

| Value | Weight VRAM | Quality | Notes |
|---|---|---|---|
| `none` | full (~46 GB) | best | the default; needs a big card |
| `fp8-scaled-mm` | ~half | very good | **and *faster* on Blackwell.** Best first move. |
| `fp8-cast` | ~half | very good | use where `scaled-mm` isn't supported |
| `nvfp4-cast` | ~quarter | lowest | Blackwell-only; smallest and fastest |
| `nvfp4-prequant` | ~quarter | lowest | as above, weights pre-quantized |

⚠️ The `nvfp4-*` modes need **Blackwell (sm_120)**. On Ampere or Ada they will
fail or fall back. `fp8-scaled-mm` needs **Hopper/Ada or newer** for real
speedups; on Ampere prefer `fp8-cast`. If in doubt, `fp8-scaled-mm` first and
fall back to `fp8-cast`.

**2. Offloading** — `SIZZLE_OFFLOAD` (default `none`)

`cpu` streams layers from pinned system RAM, `disk` from disk. Both are large
slowdowns and are a *last resort* — quantize first. `cpu` offload wants
plenty of free system RAM, so it trades one scarce resource for another.

**3. Resolution and clip length** — `SIZZLE_WIDTH` / `SIZZLE_HEIGHT` /
`SIZZLE_MAX_FRAMES`

Activation memory scales with pixels **and** with frame count (attention cost
grows with the token count). Dropping from 1280×704 to 768×448 is a big VRAM
saving and often the difference between OOM and working.

Two constraints the model enforces and you cannot ignore:

- **Both dimensions must be multiples of 64.** Stage 1 denoises at *half* the
  requested size and the upscaler doubles it, so `ltx` asserts on this.
  `validate_dimensions()` snaps custom sizes to the grid; every preset already
  complies.
- **Frame counts must be `8n+1`** (…, 449, 457, 465, …, 481 max). The temporal
  VAE compresses time 8×. `snap_frames()` enforces it — every count handed to
  LTX must go through it.

### A worked starting point for a 24 GB card

```ini
SIZZLE_QUANTIZATION=fp8-scaled-mm
SIZZLE_OFFLOAD=cpu
SIZZLE_WIDTH=768
SIZZLE_HEIGHT=448
SIZZLE_MAX_FRAMES=241        # ~10s @ 24fps, still 8n+1
SIZZLE_CACHE_WEIGHTS=0       # leave off (it caches into VRAM — see below)
```

Then walk resolution back **up** until it OOMs, and step down one notch.

### Leave `SIZZLE_CACHE_WEIGHTS` off

It sounds like an obvious win — the pipeline builds a transformer, denoises,
frees it, twice per clip. It is not. `ltx_core` loads the state dict onto the
**build** device and the registry keeps it there, so on a CUDA build this pins
~43 GB *in VRAM* on top of the model built from it. Measured on the reference
96 GB card: **96.5 of 97.9 GB resident and ~45 s/step at only 768×448.**
Lightricks' own default is `cache_weights=False` for exactly this reason. The
re-read it avoids is nearly free anyway, because the OS page cache already
holds the file. **On a smaller card this flag is strictly harmful.**

### NVENC

`SIZZLE_NVENC=1` (default) uses the GPU's hardware encoder for the final mux.
Some datacenter cards (A100, H100) have **no NVENC engine at all**, and some
consumer drivers cap concurrent sessions. If muxing fails, set
`SIZZLE_NVENC=0` to fall back to libx264 — it costs CPU time at the very end of
a job and nothing else.

### What does *not* need re-tuning

The generation defaults — `LORA_SCALE=1.2`, `IMAGE_STRENGTH=0.62`,
`A2V_GUIDANCE=3.0`, 8/20 steps, empty negative prompt — are **look**
parameters, not hardware parameters. They are Don's dialed-in settings carried
over from the hosted build and they transfer unchanged to any GPU. Changing
them changes the output, not the memory footprint. Leave them alone unless you
are deliberately re-grading.

---

## 5. Hosted / rented GPU boxes

Renting a GPU (Vast.ai, RunPod, Lambda, a cloud VM) changes four things.

**1. Put the weights on a persistent volume.** Container root disks are
ephemeral; re-downloading 128 GB on every restart is the classic and expensive
mistake.

```bash
export LTX_MODEL_ROOT=/workspace/ltx-weights     # or wherever the volume mounts
python scripts/download_models.py
```

Put that same `export` in `.env.local` so every shell and the app agree.

**2. Bind to the right interface.** The default `127.0.0.1` is correct and safe
for a local box. On a remote host you need either a tunnel (preferred) or an
explicit bind:

```ini
SIZZLE_HOST=0.0.0.0
SIZZLE_PORT=8000
```

⚠️ **`0.0.0.0` puts the app on the open internet with no authentication of any
kind.** There are no accounts and no passwords — see `MULTI-USER.md` §1: a
"user" is just a browser-generated session id. Only do this behind a firewall
or a security group that allows *your* IP. Otherwise use `host.sh`, which opens
a Cloudflare quick tunnel and gives you a random URL instead of an open port.

**3. Check the driver before you pick a torch wheel.** Hosted images often ship
a pinned driver older than the local box's. Run `nvidia-smi`, read the CUDA
version, and use `SIZZLE_TORCH_INDEX` from
[§3.1](#31--run-the-installer) if it is below 13.x.

**4. ffmpeg is often missing** on minimal CUDA images, and the stock build may
lack NVENC. `install.sh` tries apt/dnf/brew. If your image has no package
manager or you need NVENC, grab a static build:

```bash
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ
export PATH="$PWD/ffmpeg-*-amd64-static:$PATH"
```
(Static builds usually include NVENC support but still need the driver's
`libnvidia-encode` present. If in doubt, `SIZZLE_NVENC=0`.)

**Also worth knowing:** all server state is in memory (`MULTI-USER.md` §4.1),
so restarting the instance drops every in-flight job and clip. Anything you
want to keep, download from the UI first — or move `_work/` onto the
persistent volume with `SIZZLE_WORK_DIR`.

---

## 6. What is *not* in this repo, and why

Every exclusion has a documented recipe. This is the full list, so nothing is a
surprise later.

| Excluded | Why | How to get it back |
|---|---|---|
| `models/` (~123 GB) | too large for git | `scripts/download_models.py` — [`models/README.md`](models/README.md) |
| `.venv/` (~8 GB) | machine- and platform-specific | `install.bat` / `install.sh`; exact versions in `requirements.lock.txt` |
| `LTX-2/` | a clone of someone else's repo | cloned by the installers; commit pinned in `VENDOR-PINS.txt` |
| `Sizzle_upstream/` | a clone of the pre-local-GPU Sizzle, reference only | `VENDOR-PINS.txt`; **not needed to run anything** |
| `certs/ca-bundle.pem` | this box's antivirus MITM root — machine-specific and wrong elsewhere | only if you hit TLS errors: `certs/README.md` |
| `_work/` | uploads, clips, renders, logs — runtime scratch | recreated by `config.ensure_dirs()` on startup |
| `yolo.bat`, `notes.txt`, `PLAN.md` | personal working files | not needed |

The `LTX-2` exclusion is the one worth internalizing: **`backend/ltx_engine.py`
imports directly from `ltx_pipelines` and `ltx_core`.** If Lightricks change
that API upstream, a fresh clone builds a working environment that fails at
render time. That is what `VENDOR-PINS.txt` is for — pin first, debug second.

---

## 7. The `.env` in this repo — read this

**`.env` is committed here on purpose.** That is a deliberate exception to the
usual rule, made because this is a **private backup repo** and the file records
the tuned configuration and the decision to use the gated Gemma QAT copy.

**It was audited before the first commit and contains no credentials.** Its
only active line is `SIZZLE_BACKEND=local`; everything else is commented
documentation.

### The standing hazard

`.env` is now a **tracked file**. If you later paste a real key into it, a
routine `git commit -a` publishes that key to GitHub. Private repo or not,
that is how secrets leak.

> **Put secrets in `.env.local`, which is gitignored.** The two that apply
> here are `ANTHROPIC_API_KEY` and `HF_TOKEN`.

`backend/config.py` reads `.env` (or whatever `SIZZLE_ENV_FILE` points at) and
uses `os.environ.setdefault`, so **a real environment variable always wins over
the file.** That is the clean way to inject secrets on a hosted box:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
./run.sh
```

### One inconsistency you should know about

The committed `.env` is an **older copy of `.env.example`** and its comments
have drifted. Two differences, both comment-only — the live behaviour comes
from `backend/config.py`, which is correct:

- `.env` is missing the `SIZZLE_IMAGE_FIT` / `SIZZLE_IMAGE_FIT_BG` block
  (aspect-ratio letterboxing). The default is `contain`, which is what you
  want — it restores the hosted build's behaviour instead of `ltx`'s native
  centre-crop that ate the edges of mismatched images.
- `.env` describes `SIZZLE_CACHE_WEIGHTS=1` as "hugely faster; costs ~46 GB of
  RAM". **That is stale and wrong** — it caches into VRAM, not RAM, and the
  measured result was a slowdown. The default is `0` and should stay `0`. See
  [§4](#leave-sizzle_cache_weights-off).

**Treat `.env.example` as the source of truth for documentation** and `.env`
as the tuned settings file. If you want them reconciled, copy the two blocks
across — nothing functional changes either way.

---

## 8. Verification ladder

Run these in order. Each one isolates a different layer, so the first failure
tells you which step to go back to.

```bash
# 1. the GPU, the driver, and whether torch has native kernels for it
python scripts/gpu_report.py

# 2. torch actually sees CUDA (the #1 rebuild failure)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. the LTX packages import (proves the clone + editable installs worked)
python -c "from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage; print('LTX OK')"

# 4. every required weight is on disk (exit 1 if not)
python scripts/check_weights.py

# 5. the model builds and renders one short clip end to end
python scripts/test_pipeline.py

# 6. the app's own render path, outside the web server
python scripts/test_local.py

# 7. the HTTP API against a running server
python scripts/test_api.py
```

Then open the UI, drop in an image and a music file, and render one segment.
The header badge should read `render: local` — if it says
`render: local (weights missing)`, go back to step 4.

**Fastest possible smoke test with no weights at all:**
`SIZZLE_BACKEND=mock ./run.sh` — exercises every path except the model.

---

## 9. Troubleshooting the rebuild

**`torch.cuda.is_available()` is `False`**
Almost always the CPU-only wheel from PyPI. Confirm `torch.__version__` ends in
`+cu132` (or your chosen `+cuXXX`). If it doesn't, reinstall from the PyTorch
index — [§3.1](#31--run-the-installer). If it does, the driver is older than
the wheel's CUDA: use a lower `SIZZLE_TORCH_INDEX`.

**`no kernel image is available for execution on the device`**
The wheel has no native kernels for your compute capability. `gpu_report.py`
checks this explicitly. Newer card → newer CUDA index.

**`CERTIFICATE_VERIFY_FAILED` during install or download**
TLS-inspecting antivirus or a corporate proxy re-signing HTTPS with a root
Python's `certifi` doesn't trust. Only affects downloads and the Anthropic
call — local rendering needs no network. Fix: `certs/README.md` /
`scripts/fix-certs.ps1`. Note `certs/ca-bundle.pem` is gitignored precisely
because the old box's bundle is meaningless on a new one.

**`401`/`403` fetching Gemma**
The gate. Accept the licence *and* authenticate with a token from the account
that accepted it — [`models/README.md`](models/README.md#the-gemma-gate-the-one-manual-step).

**CUDA OOM on the first Generate**
Expected on anything under ~48 GB at defaults. Quantize, then shrink the
frame — [§4](#4-tuning-for-a-gpu-that-isnt-a-96-gb-blackwell). Confirm
`SIZZLE_CACHE_WEIGHTS` is not set to `1`.

**Renders fine, muxing fails**
NVENC. `SIZZLE_NVENC=0` falls back to libx264.

**Assertion about resolution or frame count**
Width/height not multiples of 64, or frames not `8n+1`. Route the value through
`config.validate_dimensions()` / `config.snap_frames()` — the UI already does.

**Import errors from `ltx_core` / `ltx_pipelines` after a fresh clone**
Upstream API drift. Check out the pinned commit in `VENDOR-PINS.txt` and
re-run the editable installs.

**It worked, then the instance restarted and everything is gone**
All server state is in memory and `_work/` is scratch. On hosted boxes set
`LTX_MODEL_ROOT` and `SIZZLE_WORK_DIR` to a persistent volume —
[§5](#5-hosted--rented-gpu-boxes).

---

## Where else to look

| Document | Covers |
|---|---|
| [`README.md`](README.md) | what Sizzle is, how it works, measured performance, tuning for look |
| [`models/README.md`](models/README.md) | the weights: manifest, download, the Gemma gate |
| [`MULTI-USER.md`](MULTI-USER.md) | what happens when you share the tunnel URL — isolation, the render lock |
| [`backend/README.md`](backend/README.md) | module-by-module tour of the server |
| [`scripts/README.md`](scripts/README.md) | every script, what it proves, when to run it |
| [`static/README.md`](static/README.md) | the front-end, and the contract it has with the API |
| [`certs/README.md`](certs/README.md) | the TLS-inspection workaround |
| [`VENDOR-PINS.txt`](VENDOR-PINS.txt) | exact upstream commits this build was validated against |
| [`.env.example`](.env.example) | every configuration knob, annotated — the source of truth |
| [`requirements.lock.txt`](requirements.lock.txt) | exact package versions from the working box |
