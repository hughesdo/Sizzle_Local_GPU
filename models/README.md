# `models/` — fetching the weights from scratch

**Nothing in this directory is in git.** It is ~123 GB of Hugging Face
checkpoints. This file is the recipe for recreating it on a new machine.

The short version:

```bash
# Windows
.venv\Scripts\python.exe scripts\download_models.py
# Linux / macOS
./.venv/bin/python scripts/download_models.py
```

That downloads everything except Gemma, which is gated — see
[The Gemma gate](#the-gemma-gate-the-one-manual-step) below.

---

## What ends up here

`backend/config.py` looks for these **exact filenames** under
`LTX_MODEL_ROOT` (default `<repo>/models`). The downloader writes them with
exactly these names, so a clean download run *is* the whole setup.

| File | Size | Source repo | Gated? | What it does |
|---|---:|---|:---:|---|
| `ltx-2.3-22b-distilled.safetensors` | 46.1 GB | `Lightricks/LTX-2.3` | no | The **Distilled (8-step, fast)** variant. This is the default and the one you want first. |
| `ltx-2.3-22b-dev.safetensors` | 46.1 GB | `Lightricks/LTX-2.3` | no | The **Full quality (slower)** variant. 20 steps, real CFG. Optional — the app runs fine with only the distilled one. |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.6 GB | `Lightricks/LTX-2.3` | no | Stage-2 refiner LoRA. **Required by both variants** — the two-stage pipeline will not build without it. Its name is misleading: it is not "the distilled model's LoRA", it is the second-stage refiner every two-stage render applies. |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 1.0 GB | `Lightricks/LTX-2.3` | no | Stage 1 denoises at half resolution; this 2× upscales the latent before stage 2. **Required.** |
| `ltx2.3_audio_reactive_lora.safetensors` | 1.35 GB | `fal/ltx2.3-audio-reactive-lora` | no | **The whole point.** This is what makes a still image move to the beat — the same adapter the hosted fal endpoint applied. |
| `ltx2.3_audio_reactive_lora_v2.safetensors` | 1.35 GB | `fal/ltx2.3-audio-reactive-lora` | no | v2 of the above, stronger beat impact / heavier motion. Opt in with `SIZZLE_LORA_V2=1`. |
| `gemma/` (directory) | 24.4 GB | `google/gemma-3-12b-it-qat-q4_0-unquantized` | **YES** | Gemma 3 12B text encoder. LTX-2.3 encodes **every** prompt through it, so renders are impossible without it. |

Total: **~128 GB downloaded, ~123 GB resident.** Budget **150 GB of free disk**
so the staging copy has room.

A `models/.cache/` directory appears during download — that is
`huggingface_hub`'s staging area. It is safe to delete once everything is in
place.

---

## The Gemma gate (the one manual step)

Google gates the Gemma repo, so this part cannot be automated. Once per
Hugging Face account:

1. **Accept the license** at
   <https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized>
   (sign in, click through the terms). This is an account-level flag — it
   follows your account to any machine.
2. **Authenticate the new box.** Either:
   ```bash
   .venv/bin/hf auth login        # interactive, stores a token in ~/.cache/huggingface
   # or
   export HF_TOKEN=hf_xxxxxxxxxxxx
   ```
   A **read**-scoped token is enough.
3. **Fetch it:**
   ```bash
   python scripts/download_models.py --only gemma
   ```

> **Where the acceptance actually lives.** It is a flag on your Hugging Face
> *account*, not a file in this repo. `.env` is committed here as a record of
> the decision to use the gated QAT copy, but it holds no token and grants no
> access. On a new machine you still have to log in with a token belonging to
> the account that accepted the licence. **Do not paste `HF_TOKEN` into
> `.env`** — it is committed. Use `.env.local` (gitignored) or the shell.

### If you would rather not accept the licence

Lightricks ship an **ungated** copy of the same Gemma 3 12B inside
[`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2), as
`text_encoder/` + `tokenizer/`. Caveats: it is stored fp32, so ~49 GB instead of
~24 GB, and it is the copy paired with their 19B model rather than the QAT
variant LTX-2.3's own docs name. Assemble it into a directory and point
`GEMMA_PATH` at it.

`scripts/download_models.py` honours `SIZZLE_GEMMA_REPO` if you want to source
Gemma from somewhere else entirely.

---

## Partial and resumed downloads

The downloader is **resumable and idempotent** — re-run it as often as you
like. It skips any file already present and non-empty, and `hf_xet` (pulled in
by `requirements.txt`) makes the large transfers much faster than plain HTTPS.

```bash
python scripts/download_models.py --list          # what's here vs missing, no download
python scripts/download_models.py --only lora     # one group at a time
python scripts/download_models.py --only distilled --only gemma
```

Groups: `distilled`, `full`, `upscaler`, `lora`, `gemma`.

**Order that gets you rendering soonest**, if you are impatient or on a slow
link (~55 GB instead of ~128 GB):

```bash
python scripts/download_models.py --only distilled --only upscaler --only lora --only gemma
```

That is a complete, working setup for the default variant. Grab `--only full`
later if you ever want the dev checkpoint.

**A half-downloaded `gemma/` is the one trap.** Its small JSON/tokenizer files
land in seconds while the five ~5 GB shards take many minutes, so a naive
"does `config.json` exist?" check says ready against a directory with no
weights in it. `config.missing_model_files()` deliberately also requires
`tokenizer.json` and at least one `model*.safetensors` shard. Trust
`scripts/check_weights.py`, not `ls`.

---

## Verifying

```bash
python scripts/check_weights.py     # exit code 1 if anything required is missing
```

`startup.bat` / `run.sh` run this automatically and warn without blocking —
the timeline editor, waveform and auto-prompt all work with no weights at all,
only **Generate** needs them. The UI badge reads `render: local (weights
missing)` until the set is complete.

To exercise the entire pipeline while the download runs, set
`SIZZLE_BACKEND=mock` — ffmpeg synthesizes each clip from the still image and
every code path except the model itself is live.

---

## Relocating the weights

Nothing requires them to live inside the repo. If the new box has a big scratch
volume:

```bash
export LTX_MODEL_ROOT=/mnt/nvme/ltx-weights
python scripts/download_models.py
```

`config.py` derives every individual path from `LTX_MODEL_ROOT`, so that one
variable moves all of them. Individual overrides exist too if you need them:
`LTX_CKPT_DISTILLED`, `LTX_CKPT_FULL`, `LTX_LORA_PATH`, `GEMMA_PATH`,
`LTX_SPATIAL_UPSCALER`, `LTX_DISTILLED_LORA`.

On a **hosted/rented GPU**, put `LTX_MODEL_ROOT` on the persistent volume, not
the container's ephemeral disk — otherwise you re-download 128 GB every time
the instance restarts. See SETUP.md § *Hosted / rented GPU boxes*.

---

## Licensing

The LTX-2.3 weights and the fal LoRA carry their own licences from their
respective Hugging Face repos, and Gemma carries Google's Gemma Terms of Use.
This project's CC BY 4.0 LICENSE covers the code in this repo only, **not** the
weights. Check each model card before doing anything commercial.
