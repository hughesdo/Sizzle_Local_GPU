"""
Fetch the local-inference weights Sizzle needs onto this box.

Sizzle renders every clip locally through Lightricks' own LTX-2 pipeline package
(A2VidPipelineTwoStage), so the same LTX-2.3 checkpoint + fal audio-reactive LoRA
that the hosted fal endpoint ran now live on your disk. Everything lands in
LTX_MODEL_ROOT (default: <repo>/models) with the filenames backend/config.py
expects, so a successful run here is all the setup the app needs.

    python scripts/download_models.py                 # everything
    python scripts/download_models.py --only gemma    # just one group
    python scripts/download_models.py --list          # show what's missing

THE GEMMA GATE
--------------
LTX-2.3 encodes prompts with Gemma 3 12B. Google gates that repo: you must accept
the Gemma license once at
    https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
and then authenticate (`hf auth login`, or HF_TOKEN=... in the environment).
Everything else here is ungated and downloads with no account at all.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(os.environ.get("LTX_MODEL_ROOT", REPO_ROOT / "models"))

LTX_REPO = "Lightricks/LTX-2.3"
FAL_LORA_REPO = "fal/ltx2.3-audio-reactive-lora"
GEMMA_REPO = os.environ.get("SIZZLE_GEMMA_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")

# (group, repo, filename, approx_gb, why it is needed)
FILES = [
    ("distilled", LTX_REPO, "ltx-2.3-22b-distilled.safetensors", 46.1,
     "the 8-step 'Distilled (fast)' variant in the UI dropdown"),
    ("distilled", LTX_REPO, "ltx-2.3-22b-distilled-lora-384.safetensors", 7.6,
     "stage-2 refiner LoRA - required by every two-stage pipeline"),
    ("full", LTX_REPO, "ltx-2.3-22b-dev.safetensors", 46.1,
     "the 'Full quality (slower)' variant in the UI dropdown"),
    ("upscaler", LTX_REPO, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", 1.0,
     "stage-1 renders at half res; this 2x upscales the latent for stage 2"),
    ("lora", FAL_LORA_REPO, "ltx2.3_audio_reactive_lora.safetensors", 1.35,
     "the audio-reactive LoRA - this is what makes clips move to the beat"),
    ("lora", FAL_LORA_REPO, "ltx2.3_audio_reactive_lora_v2.safetensors", 1.35,
     "v2 of the same LoRA (stronger beat impact); select with SIZZLE_LORA_V2=1"),
]

# Gemma is a directory, not a single file: config + weights + tokenizer assets.
# ltx_core's GemmaAssets.from_root() wants config.json + tokenizer.json alongside
# the tokenizer_config.json / processor_config.json sidecars, and
# resolve_gemma_weight_paths() globs model*.safetensors next to them.
GEMMA_PATTERNS = [
    "config.json", "generation_config.json",
    "model*.safetensors", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "processor_config.json", "preprocessor_config.json",
    "chat_template.json", "chat_template.jinja",
]
GEMMA_GB = 24.4


def _target(filename: str) -> Path:
    return MODEL_ROOT / filename


def _have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _fmt_gb(gb: float) -> str:
    return f"{gb:.1f} GB"


def report() -> list[tuple]:
    """Print what is present vs missing; return the missing file rows."""
    missing = []
    print(f"model root: {MODEL_ROOT}")
    for row in FILES:
        group, repo, filename, gb, why = row
        path = _target(filename)
        if _have(path):
            print(f"  [have] {filename:<48} {_fmt_gb(gb):>9}")
        else:
            print(f"  [MISS] {filename:<48} {_fmt_gb(gb):>9}  ({group}) {why}")
            missing.append(row)
    gemma_dir = MODEL_ROOT / "gemma"
    if (gemma_dir / "config.json").exists() and any(gemma_dir.glob("model*.safetensors")):
        print(f"  [have] gemma/{'':<42} {_fmt_gb(GEMMA_GB):>9}")
    else:
        print(f"  [MISS] gemma/{'':<42} {_fmt_gb(GEMMA_GB):>9}  (gemma) Gemma 3 12B text encoder - GATED, see below")
    return missing


def download_files(rows: list[tuple]) -> int:
    from huggingface_hub import hf_hub_download

    failed = 0
    for group, repo, filename, gb, why in rows:
        dest = _target(filename)
        if _have(dest):
            print(f"[skip] {filename} already present")
            continue
        print(f"[get ] {filename}  (~{_fmt_gb(gb)}) from {repo}\n       {why}", flush=True)
        try:
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                # local_dir means: stage into MODEL_ROOT/.cache and then MOVE
                # into place - one real copy on disk, no symlink into a separate
                # hub cache. Matters when the files are 46 GB each.
                local_dir=str(MODEL_ROOT),
            )
            print(f"[ok  ] {filename}", flush=True)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"[FAIL] {filename}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            failed += 1
    return failed


def download_gemma() -> int:
    """Fetch the Gemma 3 12B text encoder into models/gemma."""
    from huggingface_hub import snapshot_download

    dest = MODEL_ROOT / "gemma"
    if (dest / "config.json").exists() and any(dest.glob("model*.safetensors")):
        print("[skip] gemma/ already present")
        return 0
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[get ] gemma text encoder (~{_fmt_gb(GEMMA_GB)}) from {GEMMA_REPO}", flush=True)
    try:
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=str(dest),
            allow_patterns=GEMMA_PATTERNS,
            token=token,
        )
        print("[ok  ] gemma/", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] gemma: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "\n"
            "Gemma 3 12B is a GATED repository. To get it:\n"
            f"  1. Open https://huggingface.co/{GEMMA_REPO} and accept the license\n"
            "  2. Authenticate:  hf auth login       (or set HF_TOKEN=hf_...)\n"
            "  3. Re-run:        python scripts/download_models.py --only gemma\n",
            file=sys.stderr,
        )
        return 1


def main() -> int:
    groups = sorted({g for g, *_ in FILES} | {"gemma"})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=groups, default=None,
                    help="download just these groups (repeatable)")
    ap.add_argument("--list", action="store_true", help="show what is present/missing and exit")
    args = ap.parse_args()

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    missing = report()
    if args.list:
        return 0

    wanted = set(args.only) if args.only else set(groups)
    rows = [r for r in missing if r[0] in wanted]

    failed = 0
    if rows:
        print(f"\ndownloading {len(rows)} file(s)...\n")
        failed += download_files(rows)
    if "gemma" in wanted:
        failed += download_gemma()

    print("\ndone." if not failed else f"\ndone with {failed} failure(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
