"""
Central configuration for Sizzle (local-GPU edition).

Everything tweakable lives here so you are not hunting through modules.

This build renders ENTIRELY on the local NVIDIA GPU. There is no fal.ai, no
Cloudinary, and no media hosting of any kind: the LTX-2.3 checkpoint, the
audio-reactive LoRA and the Gemma text encoder all sit on this disk, and every
clip is generated in-process through Lightricks' own `ltx_pipelines` package.
The only outbound call the app can still make is the OPTIONAL Anthropic vision
auto-prompt (ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


# ----------------------------------------------------------------------------
# .env
# ----------------------------------------------------------------------------
# install.bat writes a .env and the README tells you to edit it - but nothing
# ever READ it, so every knob documented in .env.example silently did nothing
# unless you also exported it by hand. That matters a lot more now: the model
# paths, the backend, the quantization mode and your Anthropic key all live in
# environment variables.
#
# Parsed here, before anything below reads os.environ. A real environment
# variable always WINS over the file (setdefault), so `SIZZLE_BACKEND=mock
# startup.bat` still overrides a .env that says local. No dependency on
# python-dotenv - this file's format is simple enough to read in 15 lines.
def _load_dotenv(path: Path) -> None:
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            # strip one layer of matching quotes; leave inner content alone
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)
    except Exception:
        # A malformed .env must never stop the app booting - the defaults below
        # are all perfectly serviceable.
        pass


_load_dotenv(Path(os.environ.get("SIZZLE_ENV_FILE", BASE_DIR / ".env")))

# Working directories (created on startup). Kept out of the repo.
WORK_DIR = Path(os.environ.get("SIZZLE_WORK_DIR", BASE_DIR / "_work"))
UPLOAD_DIR = WORK_DIR / "uploads"      # incoming images + audio per session
CLIP_DIR = WORK_DIR / "clips"          # per-segment rendered mp4s
OUTPUT_DIR = WORK_DIR / "outputs"      # final muxed mp4s for download
LOG_DIR = Path(os.environ.get("SIZZLE_LOG_DIR", WORK_DIR / "logs"))  # admin logs
# Admin-only call ledger (JSONL). Records every Anthropic vision call (the one
# remaining paid API) and every local GPU render (free, but timed so you can see
# where the wall-clock goes). Not served to users. One JSON object per line.
API_LOG_FILE = Path(os.environ.get("SIZZLE_API_LOG", LOG_DIR / "api-calls.jsonl"))

# ----------------------------------------------------------------------------
# TLS / CA bundle  (the Avast-MITM problem)
# ----------------------------------------------------------------------------
# Avast antivirus does TLS scanning: it re-signs HTTPS with its own root, which
# lives in the Windows cert store but NOT in Python's bundled certifi. That broke
# every outbound HTTPS call with CERTIFICATE_VERIFY_FAILED and made auto-prompt
# silently fall back to one identical heuristic string.
#
# Local rendering needs no network at all, so this now only matters for the
# Anthropic auto-prompt call and for scripts/download_models.py. Resolved once at
# import, in priority order:
#   1. $SIZZLE_CA_BUNDLE            (explicit override)
#   2. certs/ca-bundle.pem          (checked-in bundle incl. the Avast root)
#   3. certifi's bundle             (fine on boxes without the MITM)
def _resolve_ca_bundle() -> str:
    override = os.environ.get("SIZZLE_CA_BUNDLE")
    if override and Path(override).exists():
        return override
    checked_in = BASE_DIR / "certs" / "ca-bundle.pem"
    if checked_in.exists():
        return str(checked_in)
    try:
        import certifi
        return certifi.where()
    except Exception:
        return ""


CA_BUNDLE = _resolve_ca_bundle()
if CA_BUNDLE:
    # Only set if unset, so an intentional external override wins.
    for _var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(_var, CA_BUNDLE)

# ----------------------------------------------------------------------------
# Model locations
# ----------------------------------------------------------------------------
# scripts/download_models.py drops everything below into LTX_MODEL_ROOT with
# exactly these filenames, so a successful download is all the setup needed.
LTX_MODEL_ROOT = Path(os.environ.get("LTX_MODEL_ROOT", BASE_DIR / "models"))

# The audio-reactive LoRA - this is what makes a still image move to the beat.
# It is the SAME adapter the hosted fal endpoint applied, now loaded locally:
#   https://huggingface.co/fal/ltx2.3-audio-reactive-lora
# v2 has stronger beat impact / heavier motion; opt in with SIZZLE_LORA_V2=1.
_LORA_FILE = ("ltx2.3_audio_reactive_lora_v2.safetensors"
              if os.environ.get("SIZZLE_LORA_V2") == "1"
              else "ltx2.3_audio_reactive_lora.safetensors")
LTX_LORA_PATH = Path(os.environ.get("LTX_LORA_PATH", LTX_MODEL_ROOT / _LORA_FILE))

# Gemma 3 12B text encoder, as a directory (config.json + model*.safetensors +
# tokenizer assets). LTX-2.3 encodes every prompt through this.
GEMMA_PATH = Path(os.environ.get("GEMMA_PATH", LTX_MODEL_ROOT / "gemma"))

# Two-stage extras. Stage 1 denoises at HALF the target resolution, then the
# spatial upscaler doubles the latent and stage 2 refines it at full size with
# the distilled LoRA applied. Both files are required by A2VidPipelineTwoStage.
SPATIAL_UPSCALER_PATH = Path(os.environ.get(
    "LTX_SPATIAL_UPSCALER",
    LTX_MODEL_ROOT / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
))
DISTILLED_LORA_PATH = Path(os.environ.get(
    "LTX_DISTILLED_LORA",
    LTX_MODEL_ROOT / "ltx-2.3-22b-distilled-lora-384.safetensors",
))
DISTILLED_LORA_STRENGTH = float(os.environ.get("LTX_DISTILLED_LORA_STRENGTH", 1.0))

# Variant checkpoints. The user toggles between these per session (the dropdown
# in the header). Both are 22B monolith checkpoints bundling the transformer,
# the video + audio VAEs and the text projection in one file.
VARIANTS = {
    "distilled": {
        "label": "Distilled (8-step, fast)",
        "checkpoint": os.environ.get(
            "LTX_CKPT_DISTILLED",
            str(LTX_MODEL_ROOT / "ltx-2.3-22b-distilled.safetensors"),
        ),
        "pipeline": "distilled",   # 8 baked-in sigmas, CFG=1
    },
    "full": {
        "label": "Full quality (slower)",
        "checkpoint": os.environ.get(
            "LTX_CKPT_FULL",
            str(LTX_MODEL_ROOT / "ltx-2.3-22b-dev.safetensors"),
        ),
        "pipeline": "full",        # scheduled sigmas + real CFG
    },
}
DEFAULT_VARIANT = "distilled"

# ----------------------------------------------------------------------------
# Render backend
# ----------------------------------------------------------------------------
# "local" -> drive LTX-2.3 in-process on this box's GPU (the default, and the
#            whole point of this build). Needs the weights above + CUDA.
# "mock"  -> no model at all: synthesize each clip locally with ffmpeg from the
#            first-frame image (slow zoom + beat-tinted overlay + frame counter).
#            Needs no weights and no GPU. Lets you build/preview timelines and
#            exercise the FULL pipeline (continues, last-frame carry-over, black
#            fill, partial-window mux) while the 100 GB of weights download.
RENDER_BACKEND = os.environ.get("SIZZLE_BACKEND", "local")

# Sampler budget per variant. The distilled checkpoint ships an 8-step sigma
# schedule and wants CFG=1; the dev checkpoint uses the scheduler with real
# guidance. These mirror the step counts the hosted endpoint used (8 / 20).
LOCAL_STEPS = {
    "distilled": int(os.environ.get("SIZZLE_STEPS_DISTILLED", 8)),
    "full": int(os.environ.get("SIZZLE_STEPS_FULL", 20)),
}

# Classifier-free guidance for the video stream. The distilled model is trained
# to run at CFG 1 (guidance is baked in); the dev model wants ~3.
CFG_SCALE = {
    "distilled": float(os.environ.get("SIZZLE_CFG_DISTILLED", 1.0)),
    "full": float(os.environ.get("SIZZLE_CFG_FULL", 3.0)),
}
# Spatio-temporal guidance: a quality boost that costs an extra model pass per
# step. Off for distilled (CFG=1 leaves nothing to skip), on for dev.
STG_SCALE = {
    "distilled": float(os.environ.get("SIZZLE_STG_DISTILLED", 0.0)),
    "full": float(os.environ.get("SIZZLE_STG_FULL", 1.0)),
}
STG_BLOCKS = [int(b) for b in os.environ.get("SIZZLE_STG_BLOCKS", "28").split(",") if b.strip()]
RESCALE_SCALE = float(os.environ.get("SIZZLE_RESCALE", 0.7))
# How hard the audio track steers the video. This is the audio->video modality
# guidance; turn it UP for more literal beat-locking, DOWN for looser motion.
A2V_GUIDANCE = float(os.environ.get("SIZZLE_A2V_GUIDANCE", 3.0))

# LTX ships a long default negative prompt (blur, artifacts, deformed hands...).
# The fal LoRA card recommends an EMPTY negative prompt unless you need
# constraints, and at CFG=1 the negative branch is not evaluated at all.
NEGATIVE_PROMPT = os.environ.get("SIZZLE_NEGATIVE_PROMPT", "")

# ----------------------------------------------------------------------------
# GPU / memory
# ----------------------------------------------------------------------------
# Quantization for the 22B transformer. On a 96 GB card bf16 (none) fits with
# room to spare and is the highest quality, so that is the default here.
# Options: none | fp8-cast | fp8-scaled-mm | nvfp4-cast | nvfp4-prequant
#   fp8-scaled-mm  ~halves weight VRAM and is FASTER on Blackwell
#   nvfp4-*        smallest + fastest on Blackwell, lowest fidelity
QUANTIZATION = os.environ.get("SIZZLE_QUANTIZATION", "none").strip().lower()

# Weight offloading. "none" keeps everything resident (fastest, needs the VRAM);
# "cpu" streams layers from pinned RAM; "disk" streams from disk. A 96 GB card
# never needs to offload.
OFFLOAD_MODE = os.environ.get("SIZZLE_OFFLOAD", "none").strip().lower()

# Cache the raw checkpoint tensors in the loader's registry between builds.
#
# OFF BY DEFAULT, and the reason matters. The pipeline builds a transformer,
# denoises, then frees it - twice per clip (stage 1 and stage 2), because the
# two stages carry different LoRA sets. Caching the state dict sounds like an
# obvious win for that. It is not: ltx_core loads the state dict onto the BUILD
# device and the registry keeps it there, so on a CUDA build this caches ~43 GB
# in VRAM, not in system RAM. You then hold the cached copy AND the fused model
# at once - ~84 GB on a 96 GB card - which leaves almost nothing for
# activations and makes rendering dramatically slower, or OOMs outright.
# (Measured on the RTX PRO 6000: 96.5 GB of 97.9 GB resident and ~45 s/step at
# only 768x448, versus far less VRAM and much faster with it off.)
#
# The re-read it avoids is largely free anyway: the OS page cache keeps the
# 43 GB file resident in spare RAM after the first read, so the second stage
# reads from memory without us paying for a second copy in VRAM. That is the
# real "cache the weights in RAM" mechanism, and it needs no flag.
#
# Lightricks' own default is cache_weights=False for exactly this reason. Turn
# this on only if you have VRAM to burn and have measured a win.
CACHE_WEIGHTS_IN_RAM = os.environ.get("SIZZLE_CACHE_WEIGHTS", "0") == "1"

# ----------------------------------------------------------------------------
# Generation defaults (Don's dialed-in settings, carried over from the fal build)
# ----------------------------------------------------------------------------
FPS = 24
LORA_SCALE = float(os.environ.get("SIZZLE_LORA_SCALE", 1.2))  # LoRA card: 1.2-1.5
IMAGE_STRENGTH = float(os.environ.get("SIZZLE_IMAGE_STRENGTH", 0.62))  # first-frame conditioning

# How a dropped image is mapped onto the output frame when their aspect ratios
# differ.
#
#   contain  the WHOLE image is kept, letterboxed with black to fill the frame.
#   cover    the frame is filled edge to edge and the overflow is cropped away,
#            so parts of the image are lost.
#   stretch  the image is distorted to exactly fit the frame.
#
# This used to be fal's job. Upstream did no image processing at all - it handed
# the endpoint a URL and fal fitted the picture server-side, keeping the whole
# image. Rendering locally, ltx's own conditioning path instead does a
# resize-and-CENTER-CROP (see resize_and_center_crop: scale = max(...)), which
# silently ate the edges of every image that was not already the output aspect.
# `contain` restores the behaviour the hosted build had.
IMAGE_FIT = os.environ.get("SIZZLE_IMAGE_FIT", "contain").strip().lower()
# Colour of the letterbox bars for `contain`, as an R,G,B triple.
IMAGE_FIT_BG = tuple(
    int(c) for c in os.environ.get("SIZZLE_IMAGE_FIT_BG", "0,0,0").split(",")
)[:3]

# ----------------------------------------------------------------------------
# LTX-2.3 frame-count constraint  (the "why the cap?" ceiling + sync guarantee)
# ----------------------------------------------------------------------------
# LTX-2.3's temporal VAE compresses time 8x, so the model ONLY accepts frame
# counts F where (F-1) % 8 == 0  ->  F = 8n+1:
#     ... 105, 113, 121, 129 ... 449, 457, 465, 473, 481.
# Locally there is no endpoint slider to hit, but the cap still earns its keep:
# attention cost grows with the token count, so a very long single generation is
# both slow and VRAM-hungry. The fal-era ceiling is kept as the default so
# timelines built on the hosted build still render identically here.
FRAME_ALIGN = 8
HARD_MAX_FRAMES = 481                     # 20.04s @ 24fps

MAX_FRAMES = int(os.environ.get("SIZZLE_MAX_FRAMES", 457))   # 19.04s @ 24fps
# Floor: LTX does poorly on very short, low-motion clips. 49 = 8*6+1 = 2.04s.
MIN_FRAMES = int(os.environ.get("SIZZLE_MIN_FRAMES", 49))


def snap_frames(frames: int) -> int:
    """Snap an arbitrary frame count to the nearest valid LTX value (F=8n+1),
    clamped to [MIN_FRAMES, MAX_FRAMES]. EVERY count handed to LTX must pass
    through here, or the latent grid does not line up."""
    f = round((int(frames) - 1) / FRAME_ALIGN) * FRAME_ALIGN + 1
    return max(MIN_FRAMES, min(MAX_FRAMES, f))


def frames_for(seconds: float) -> int:
    """The valid LTX frame count closest to `seconds` of footage at FPS."""
    return snap_frames(round(seconds * FPS))


def seconds_for(frames: int) -> float:
    """Exact real duration a frame count occupies at FPS. This is the length the
    audio must be trimmed to so video and audio stay sample-accurate in the mux."""
    return frames / FPS


# Derived clip-length caps (seconds). The whole app + front-end reason in these;
# they come straight from the frame ceiling so the UI can never place a block the
# model would reject. MAX_CLIP_SECONDS = 457/24 = 19.04s by default.
MAX_CLIP_SECONDS = seconds_for(MAX_FRAMES)
MIN_CLIP_SECONDS = seconds_for(MIN_FRAMES)

# ----------------------------------------------------------------------------
# Output resolution
# ----------------------------------------------------------------------------
# The UI can pick any preset below or a custom size per job.
#
# HARD CONSTRAINT: the two-stage pipeline denoises stage 1 at HALF the requested
# size and then upscales 2x, so BOTH dimensions must be divisible by 64 (ltx
# asserts this - see assert_resolution(is_two_stage=True)). The hosted endpoint
# was permissive about this and quietly fitted odd sizes; running the model
# ourselves, we have to be exact. Every preset below is a multiple of 64, and
# validate_dimensions() snaps custom sizes to the 64 grid.
DIM_ALIGN = 64

WIDTH = int(os.environ.get("SIZZLE_WIDTH", 1280))
HEIGHT = int(os.environ.get("SIZZLE_HEIGHT", 704))   # landscape default

MIN_DIM = int(os.environ.get("SIZZLE_MIN_DIM", 256))    # 64*4
MAX_DIM = int(os.environ.get("SIZZLE_MAX_DIM", 1920))   # 64*30

# Curated format presets, site-labeled + intuitive. Sizes are the nearest 64-grid
# neighbours of the originals, so the aspect ratios the UI offers are unchanged
# to the eye (720 -> 704, 1080 -> 1088).
FORMAT_PRESETS = [
    {"id": "portrait",     "label": "Portrait 9:16 - TikTok / Reels / Shorts", "width": 768,  "height": 1280},
    {"id": "portrait_hd",  "label": "Portrait 9:16 - HD (slower)",             "width": 1088, "height": 1920},
    {"id": "landscape",    "label": "Landscape 16:9 - YouTube",                "width": 1280, "height": 704},
    {"id": "landscape_hd", "label": "Landscape 16:9 - HD (slower)",            "width": 1920, "height": 1088},
    {"id": "square",       "label": "Square 1:1 - Instagram",                  "width": 960,  "height": 960},
    {"id": "square_hd",    "label": "Square 1:1 - Instagram HD",               "width": 1088, "height": 1088},
    {"id": "wide",         "label": "Ultrawide 2.39:1 - cinematic",            "width": 1664, "height": 704},
]


def megapixels(width: int, height: int) -> float:
    """Frame size in megapixels. Local renders cost no money, but they do cost
    TIME, and time scales with pixels - so this is what the format dropdown
    shows instead of the old dollars-per-second."""
    return round(width * height / 1_000_000.0, 2)


def presets_payload() -> list:
    """Preset list for the frontend, each with its frame megapixels so the
    dropdown can hint at the relative render cost of the choice."""
    return [{**p, "megapixels": megapixels(p["width"], p["height"])}
            for p in FORMAT_PRESETS]


def validate_dimensions(width, height) -> tuple[int, int]:
    """Coerce + bound-check a requested WxH, snapping to the 64-pixel grid the
    two-stage pipeline requires. Raises ValueError on anything out of range or
    non-integer. Snapping (rather than rejecting) keeps the custom-size box
    forgiving: type 1000 and you get 1024, instead of a 422."""
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        raise ValueError("width/height must be integers")
    for name, v in (("width", w), ("height", h)):
        if v < MIN_DIM or v > MAX_DIM:
            raise ValueError(f"{name} {v} out of range [{MIN_DIM}, {MAX_DIM}]")

    def _snap(v: int) -> int:
        s = round(v / DIM_ALIGN) * DIM_ALIGN
        return max(MIN_DIM, min(MAX_DIM, s))

    return _snap(w), _snap(h)


# ----------------------------------------------------------------------------
# ffmpeg / final encode
# ----------------------------------------------------------------------------
# NVENC for speed since you are already on the GPU; libx264 fallback.
USE_NVENC = os.environ.get("SIZZLE_NVENC", "1") == "1"
# Don's X-safe mux flags for the audio track.
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
FASTSTART_ARGS = ["-movflags", "+faststart"]

# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
HOST = os.environ.get("SIZZLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SIZZLE_PORT", 8000))


# ----------------------------------------------------------------------------
# Render resilience
# ----------------------------------------------------------------------------
# If a segment's render fails, swap in a different image for that slot and try
# again, so the mux stays gapless and the audio never desyncs. Locally the common
# cause is no longer a content-policy rejection (there is no filter in the loop
# any more) but a genuine CUDA OOM or a bad input image. This caps total
# render_segment calls per segment (original attempt + substitutes).
RENDER_MAX_ATTEMPTS = int(os.environ.get("SIZZLE_RENDER_MAX_ATTEMPTS", 4))


def ensure_dirs() -> None:
    for d in (WORK_DIR, UPLOAD_DIR, CLIP_DIR, OUTPUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def missing_model_files(variant: str | None = None) -> list:
    """Which required weights are absent, as (label, path) pairs. Drives the
    'render: local (weights missing)' badge and the startup hint, so a fresh
    clone tells you to run scripts/download_models.py instead of exploding on
    the first Generate.

    `variant` scopes the CHECKPOINT check to one variant. That matters because
    the two 22B checkpoints are ~46 GB each and download one after the other:
    the distilled variant is perfectly renderable while the dev one is still
    coming down, and reporting the whole app as unusable for an hour because a
    checkpoint the user has not selected is absent is just wrong. Pass None to
    audit everything (what the downloader and the readiness script want).
    """
    missing = []
    wanted = {variant} if variant else set(VARIANTS)
    for key, v in VARIANTS.items():
        if key not in wanted:
            continue
        p = Path(v["checkpoint"])
        if not p.exists():
            missing.append((f"{key} checkpoint", str(p)))
    for label, p in (
        ("audio-reactive LoRA", LTX_LORA_PATH),
        ("spatial upscaler", SPATIAL_UPSCALER_PATH),
        ("distilled LoRA", DISTILLED_LORA_PATH),
    ):
        if not Path(p).exists():
            missing.append((label, str(p)))
    # Gemma is a DIRECTORY, and a partial download is the dangerous case: the
    # small json/tokenizer files land in seconds while the five ~5 GB weight
    # shards take many minutes, so testing config.json alone reports "ready"
    # against a directory with no weights in it at all. Require what
    # ltx_core actually loads: the config, the tokenizer, and at least one
    # model*.safetensors shard (resolve_gemma_weight_paths globs for exactly
    # that, and GemmaAssets.from_root needs the other two).
    gemma_missing = [
        name for name in ("config.json", "tokenizer.json")
        if not (GEMMA_PATH / name).exists()
    ]
    if not any(GEMMA_PATH.glob("model*.safetensors")):
        gemma_missing.append("model*.safetensors")
    if gemma_missing:
        missing.append((f"gemma text encoder ({', '.join(gemma_missing)})",
                        str(GEMMA_PATH)))
    return missing


# ----------------------------------------------------------------------------
# Cost estimation (for the admin call log)
# ----------------------------------------------------------------------------
# Anthropic list prices, $ per token (input, output). The auto-prompt vision call
# is now the ONLY paid API in the app, so this table is the whole cost model.
ANTHROPIC_PRICING = {
    "claude-opus-5":     (5.00 / 1e6, 25.00 / 1e6),
    "claude-opus-4-8":   (5.00 / 1e6, 25.00 / 1e6),
    "claude-sonnet-5":   (3.00 / 1e6, 15.00 / 1e6),
    "claude-haiku-4-5":  (1.00 / 1e6,  5.00 / 1e6),
}


def anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated $ for one Anthropic call; None if the model isn't in the table."""
    price = ANTHROPIC_PRICING.get(model)
    if not price or input_tokens is None or output_tokens is None:
        return None
    pin, pout = price
    return round(input_tokens * pin + output_tokens * pout, 6)
