# `scripts/` — setup, diagnostics, and the verification ladder

Every script is standalone and safe to run at any time. Together they are how
you prove a rebuild on a new GPU actually works, layer by layer.

Paths below use `python` for brevity — use `.venv\Scripts\python.exe` on
Windows or `./.venv/bin/python` elsewhere.

## Setup

### `download_models.py` — fetch the ~128 GB of weights
```bash
python scripts/download_models.py                 # everything
python scripts/download_models.py --list          # what's here vs missing
python scripts/download_models.py --only gemma    # one group at a time
```
Resumable and idempotent; skips anything already present. Groups: `distilled`,
`full`, `upscaler`, `lora`, `gemma`. Honours `LTX_MODEL_ROOT` (put it on the
persistent volume on a hosted box) and `SIZZLE_GEMMA_REPO`.

**Gemma is gated** — accept the licence once, then `hf auth login` or set
`HF_TOKEN`. Full detail: [`../models/README.md`](../models/README.md).

### `fix-certs.ps1` — only if HTTPS fails (Windows)
Appends the Windows trusted roots into certifi's bundle. Needed on boxes where
antivirus or a corporate proxy does TLS inspection and every download dies with
`CERTIFICATE_VERIFY_FAILED`. Idempotent — keeps a `.orig` and rebuilds from it,
so re-running never double-appends. **Re-run it after any `pip install certifi`
or Python upgrade**, which silently overwrite the patched bundle.

Local rendering needs no network at all, so this only ever affects downloads and
the Anthropic auto-prompt. See [`../certs/README.md`](../certs/README.md).

## Diagnostics

### `gpu_report.py` — what GPU is *really* here
```bash
python scripts/gpu_report.py     # or GPUQuery.bat on Windows
```
**Run this first on any new box.** Reports real VRAM, compute capability,
whether your torch build has *native* kernels for that capability (rather than
JIT-ing from PTX on every start), and a verdict on bf16 vs fp8 vs NVFP4.

It exists because Windows lies: `Win32_VideoController.AdapterRAM` is a 32-bit
field, so any card ≥ 4 GiB wraps to ~4.29 GB — and Device Manager, dxdiag and
Task Manager all read it. The script asks the driver and the CUDA runtime
instead and prints Windows' wrong answer alongside, so the discrepancy reads as
a known quirk rather than a failure.

`GPUWatch.bat` (repo root) is the live companion — watch VRAM and utilization
while a render runs, which is how you find the resolution ceiling for a card.

### `check_weights.py` — is this box ready to render?
```bash
python scripts/check_weights.py           # human-readable
python scripts/check_weights.py --quiet   # exit code only
```
Exit 0 if ready, 1 if anything is missing. The launchers call it and warn
without blocking. Checks the **whole** set — checkpoint *and* both LoRAs *and*
the upscaler *and* Gemma — because missing any one of them fails at Generate
time rather than at startup. It also catches the half-downloaded `gemma/` case,
where the JSONs have landed but the 5 GB shards haven't.

## Tests — the verification ladder

Run in this order after a rebuild. Each isolates a different layer, so the
first failure tells you which step to go back to.

| # | Script | Needs GPU? | Needs weights? | Proves |
|---|---|:---:|:---:|---|
| 1 | `gpu_report.py` | ✅ | ❌ | driver, VRAM, kernel coverage |
| 2 | `check_weights.py` | ❌ | — | every required file is on disk |
| 3 | `test_pipeline.py` | ❌ | ❌ | timeline logic: continues, gap fill, mux |
| 4 | `test_local.py` | ✅ | ✅ | the model renders a real clip |
| 5 | `test_api.py` | ✅ | ✅ | the HTTP + WebSocket surface |

### `test_pipeline.py` — the pipeline without the GPU
```bash
SIZZLE_BACKEND=mock python scripts/test_pipeline.py
```
Builds a fake timeline (image → continue → gap → image), runs the job loop
synchronously with no server and no queue, then ffprobes the output. Proves
last-frame extraction, continue chaining, black gap fill and the partial-window
audio mux — **all without touching the GPU or the weights.**

This is the best first test on a fresh box. If it passes, your Python
environment and ffmpeg are correct and only the model layer is unverified.

### `test_local.py` — the real GPU render path, no web server
```bash
python scripts/test_local.py                              # synthetic image + beat track
python scripts/test_local.py --image my.jpg --audio song.wav
python scripts/test_local.py --seconds 3 --width 768 --height 448
python scripts/test_local.py --variant full
```
Drives `backend.ltx_engine` directly. Reports wall-clock, realtime factor and
**peak VRAM**, then ffprobes the output so a clip that decodes to the wrong
frame count fails loudly instead of looking fine.

**This is the script for dialling in a new card.** Start small
(`--width 768 --height 448 --seconds 3`), watch peak VRAM, and walk up until it
OOMs. That number is your ceiling — see `SETUP.md` §4.

### `test_api.py` — end to end over HTTP
```bash
python scripts/test_api.py     # against an already-running server on 127.0.0.1:8000
```
Drives the real HTTP + WS surface the browser uses: audio upload → waveform,
beats, downbeats, tempo; autoprompt variation per image (a **real** Anthropic
call, so it needs `ANTHROPIC_API_KEY`); then a full `/api/generate` with an
image + gap + continue block, capturing WS progress and ffprobing the download.

## Rebuild notes

- Scripts add the repo root to `sys.path` themselves, so run them from
  anywhere — but `LTX_MODEL_ROOT` still defaults relative to the repo.
- `test_api.py` needs a live server (`startup.bat` / `./run.sh`) in another
  window, and an Anthropic key for the autoprompt assertions.
- `download_models.py` is the only script that hits the network for large
  files; if it fails on TLS, that is `fix-certs.ps1` territory, not a bug.
- `fix-certs.ps1` is Windows/PowerShell only and is deliberately not ported —
  on Linux the equivalent is `export SSL_CERT_FILE=/path/to/bundle.pem`, which
  `config.py` already honours via `SIZZLE_CA_BUNDLE`.
