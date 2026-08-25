"""
LTX-2.3 audio-to-video engine - LOCAL GPU.

This drives Lightricks' own inference package (Lightricks/LTX-2, the
`ltx-pipelines` package) in-process on this box's NVIDIA GPU. NO ComfyUI, NO
fal.ai, NO media hosting. The pipeline is `A2VidPipelineTwoStage`: audio-to-video
conditioned on an audio slice, with first-frame image conditioning and the fal
audio-reactive LoRA applied on top - the same model + adapter combination the
hosted endpoint ran, now entirely local.

How a clip is made
------------------
  stage 1   denoise video at HALF the target resolution, with the audio latent
            FROZEN as conditioning (the audio drives motion; it is never
            regenerated)
  upsample  2x the video latent with the spatial upscaler
  stage 2   refine at full resolution with the distilled LoRA applied
  decode    VAE-decode to RGB frames, encode to mp4 (video only)

The mp4 written here is deliberately SILENT. The generated audio is discarded:
mux.py lays the ORIGINAL continuous track over the concatenated timeline so the
music never breaks at a clip seam. (This is what `generate_audio: False` bought
us on the hosted endpoint.)

Memory model
------------
`DiffusionStage` builds a transformer, denoises, then frees it - so peak VRAM is
ONE 22B transformer, not two, even though the two stages carry different LoRA
sets. The cost is a checkpoint re-read between stages, which the OS page cache
absorbs (the 43 GB file stays resident in spare system RAM after the first
read). We deliberately do NOT cache the state dict in the loader registry:
ltx_core keeps it on the BUILD device, so that would pin ~43 GB in VRAM on top
of the built model. See config.CACHE_WEIGHTS_IN_RAM.

The module is import-safe even without torch/ltx installed: heavy imports are
deferred into _lazy_import so the web app can boot and serve the UI on a machine
that isn't the GPU box (useful while iterating on the front-end, or while the
weights are still downloading).
"""
from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

from . import config

# Module-level singletons: load the 22B model ONCE and keep it warm for the whole
# session. Rebuilding per clip would be brutal.
_PIPELINE = None
_PIPELINE_VARIANT: Optional[str] = None
_REGISTRY = None
_LOAD_LOCK = threading.Lock()


class LtxNotInstalled(RuntimeError):
    pass


def _lazy_import():
    """Import torch + the LTX packages only when we actually render."""
    try:
        import torch  # noqa: F401
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage  # type: ignore
        return torch, A2VidPipelineTwoStage
    except Exception as e:  # ImportError or nested import failures
        raise LtxNotInstalled(
            "LTX-2 inference packages not importable. Install them into this "
            "venv (see install.bat / README) so `ltx_pipelines` imports. "
            f"Original error: {e}"
        ) from e


def _quantization_policy(checkpoint: str):
    """Build the QuantizationPolicy named by SIZZLE_QUANTIZATION, or None for
    plain bf16. bf16 is the default because a 96 GB card has the headroom and it
    is the highest-fidelity option; fp8/nvfp4 trade a little quality for VRAM
    and (on Blackwell) speed."""
    kind = (config.QUANTIZATION or "none").strip().lower()
    if kind in ("", "none", "bf16", "off"):
        return None
    from ltx_pipelines.utils.quantization_factory import QuantizationKind  # type: ignore
    try:
        return QuantizationKind(kind).to_policy(checkpoint)
    except ValueError as e:
        valid = ", ".join(k.value for k in QuantizationKind)
        raise RuntimeError(
            f"SIZZLE_QUANTIZATION={kind!r} is not a known policy. Use one of: none, {valid}"
        ) from e


def _offload_mode():
    from ltx_pipelines.utils.types import OffloadMode  # type: ignore
    try:
        return OffloadMode(config.OFFLOAD_MODE or "none")
    except ValueError:
        valid = ", ".join(m.value for m in OffloadMode)
        raise RuntimeError(
            f"SIZZLE_OFFLOAD={config.OFFLOAD_MODE!r} is not valid. Use one of: {valid}"
        )


def _registry():
    """Shared weight/shell cache handed to every component of the pipeline.

    cache_models=True reuses the model SHELL (structure only, cheap) across
    builds. cache_weights follows config.CACHE_WEIGHTS_IN_RAM and defaults to
    False: ltx_core loads state dicts onto the build device, so caching them
    would hold ~43 GB in VRAM alongside the model that was built from them.
    When it IS enabled the loader stays correct - it detects a retained cache
    entry and fuses LoRAs into a fresh copy rather than mutating the cache - it
    is purely a memory decision.
    """
    global _REGISTRY
    if _REGISTRY is None:
        from ltx_core.loader.registry import ModelRegistry  # type: ignore
        _REGISTRY = ModelRegistry(
            cache_weights=config.CACHE_WEIGHTS_IN_RAM,
            cache_models=True,
        )
    return _REGISTRY


def _fit_image(image_path: Path, width: int, height: int) -> Path:
    """Map the conditioning image onto an exactly width x height canvas.

    Why this exists: upstream Sizzle did no image processing whatsoever - it
    handed fal a URL and fal fitted the picture on its own servers, keeping the
    whole image. Running the model ourselves, ltx's conditioning path applies
    `resize_and_center_crop` (scale = max(h/src_h, w/src_w)), which fills the
    frame and CROPS AWAY whatever overflows. Any image that is not already the
    output aspect ratio silently loses its edges.

    Producing an image that is already exactly the target size makes ltx's own
    resize a no-op, so this is the only place the decision is made.

    Returns the original path untouched when no work is needed (already the
    right size, or SIZZLE_IMAGE_FIT=cover, which is ltx's native behaviour).
    """
    mode = (config.IMAGE_FIT or "contain").strip().lower()
    if mode == "cover":
        return image_path  # let ltx do exactly what it does by default

    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if im.size == (width, height):
            return image_path

        if mode == "stretch":
            canvas = im.resize((width, height), Image.LANCZOS)
        else:  # contain - keep the whole image, pad the remainder
            if mode != "contain":
                raise ValueError(
                    f"SIZZLE_IMAGE_FIT={mode!r} is not valid. "
                    "Use one of: contain, cover, stretch."
                )
            scale = min(width / im.width, height / im.height)
            new = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            resized = im.resize(new, Image.LANCZOS)
            bg = tuple(config.IMAGE_FIT_BG) if len(config.IMAGE_FIT_BG) == 3 else (0, 0, 0)
            canvas = Image.new("RGB", (width, height), bg)
            canvas.paste(resized, ((width - new[0]) // 2, (height - new[1]) // 2))

        # Keep fitted copies beside the clips so they are cleaned up with the
        # rest of a job's scratch, and never touch the user's upload.
        out = config.CLIP_DIR / f"fit_{mode}_{width}x{height}_{image_path.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
        return out


def _require_weights(variant: str | None = None) -> None:
    """Fail fast, and only on weights this variant actually needs."""
    missing = config.missing_model_files(variant)
    if missing:
        lines = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise LtxNotInstalled(
            "Local model weights are missing:\n" + lines +
            "\n\nRun:  python scripts/download_models.py"
        )


def _build_pipeline(variant: str):
    """Construct A2VidPipelineTwoStage for `variant`, with the audio-reactive
    LoRA applied to both stages and the distilled LoRA added on stage 2."""
    torch, A2VidPipelineTwoStage = _lazy_import()
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps  # type: ignore
    from ltx_pipelines.utils.model_paths import ModelPaths  # type: ignore

    _require_weights(variant)
    vconf = config.VARIANTS[variant]
    checkpoint = vconf["checkpoint"]

    # Monolith layout: one fat .safetensors carrying the transformer, both VAEs
    # and the text projection, plus the Gemma directory alongside it.
    model_paths = ModelPaths.from_monolith(checkpoint, str(config.GEMMA_PATH))

    # The audio-reactive LoRA rides on BOTH stages (the pipeline appends the
    # distilled LoRA to this set for stage 2). Comfy key naming is what these
    # community LoRAs ship with, which is why LTXV_LORA_COMFY_RENAMING_MAP is
    # the sd_ops the CLI uses too.
    loras = [LoraPathStrengthAndSDOps(
        str(config.LTX_LORA_PATH), config.LORA_SCALE, LTXV_LORA_COMFY_RENAMING_MAP)]
    distilled_lora = [LoraPathStrengthAndSDOps(
        str(config.DISTILLED_LORA_PATH), config.DISTILLED_LORA_STRENGTH,
        LTXV_LORA_COMFY_RENAMING_MAP)]

    pipe = A2VidPipelineTwoStage(
        model_paths=model_paths,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(config.SPATIAL_UPSCALER_PATH),
        loras=loras,
        device=torch.device("cuda"),
        quantization=_quantization_policy(checkpoint),
        registry=_registry(),
        offload_mode=_offload_mode(),
    )
    return pipe


def get_pipeline(variant: str):
    """Return the warm pipeline, rebuilding only if the variant changed."""
    global _PIPELINE, _PIPELINE_VARIANT
    with _LOAD_LOCK:
        if _PIPELINE is not None and _PIPELINE_VARIANT == variant:
            return _PIPELINE
        # variant switch: drop the old one (and its cached 46 GB of tensors)
        # before loading the new one, or RAM grows by a whole checkpoint.
        if _PIPELINE is not None:
            _free_pipeline()
        _PIPELINE = _build_pipeline(variant)
        _PIPELINE_VARIANT = variant
        return _PIPELINE


def _free_pipeline():
    global _PIPELINE, _PIPELINE_VARIANT, _REGISTRY
    try:
        if _REGISTRY is not None:
            _REGISTRY.clear()
    except Exception:
        pass
    _REGISTRY = None
    _PIPELINE = None
    _PIPELINE_VARIANT = None
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def warmup(variant: Optional[str] = None) -> None:
    """Build the pipeline ahead of the first Generate. Optional: without it the
    first clip of a session pays the model-load cost inline."""
    if config.RENDER_BACKEND != "local":
        return
    get_pipeline(variant or config.DEFAULT_VARIANT)


def _guider_params(variant: str):
    """Video-stream guidance for this variant.

    distilled: CFG 1 / STG off - guidance is baked into the distilled weights,
               so a second (negative) model pass would only cost time.
    full:      real CFG + spatio-temporal guidance, LTX-2.3's own defaults.
    `modality_scale` is the audio->video term: how hard the track steers motion.
    """
    from ltx_core.components.guiders import MultiModalGuiderParams  # type: ignore
    return MultiModalGuiderParams(
        cfg_scale=config.CFG_SCALE.get(variant, 1.0),
        stg_scale=config.STG_SCALE.get(variant, 0.0),
        rescale_scale=config.RESCALE_SCALE,
        modality_scale=config.A2V_GUIDANCE,
        skip_step=0,
        stg_blocks=list(config.STG_BLOCKS),
    )


def _stage_1_sigmas(variant: str):
    """The distilled checkpoint ships a fixed 8-step sigma schedule; handing it
    the generic scheduler instead is what makes distilled output look washed
    out. The dev checkpoint uses the scheduler (num_inference_steps)."""
    if config.VARIANTS[variant]["pipeline"] != "distilled":
        return None
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS  # type: ignore
    return DISTILLED_SIGMAS


def render_segment(
    *,
    variant: str,
    image_path: Path,
    audio_slice_path: Path,
    prompt: str,
    num_frames: int,
    out_path: Path,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    job_id: Optional[str] = None,
) -> Path:
    """
    Render one timeline segment to out_path (silent mp4, no final mux yet).

    image_path       first-frame conditioning image
    audio_slice_path the audio for THIS segment (drives the reactive motion)
    prompt           the per-image prompt (typed or auto-generated)
    num_frames       duration_seconds * fps, already snapped to F=8n+1
    width/height     render resolution for this job; None falls back to the
                     config default (SIZZLE_WIDTH/HEIGHT). One size per job.
    """
    w = width or config.WIDTH
    h = height or config.HEIGHT

    if config.RENDER_BACKEND == "mock":
        return _render_mock(
            image_path=image_path,
            prompt=prompt,
            num_frames=num_frames,
            out_path=out_path,
            width=w,
            height=h,
        )

    return _render_local(
        variant=variant,
        image_path=image_path,
        audio_slice_path=audio_slice_path,
        prompt=prompt,
        num_frames=num_frames,
        out_path=out_path,
        seed=seed,
        width=w,
        height=h,
        job_id=job_id,
    )


def _render_local(
    *,
    variant: str,
    image_path: Path,
    audio_slice_path: Path,
    prompt: str,
    num_frames: int,
    out_path: Path,
    seed: Optional[int],
    width: int,
    height: int,
    job_id: Optional[str] = None,
) -> Path:
    """Generate one clip on the local GPU and write it to out_path."""
    from . import apilog
    torch, _ = _lazy_import()
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number  # type: ignore
    from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore
    from ltx_pipelines.utils.media_io import encode_video  # type: ignore

    # Both constraints are enforced at the edges too, but a stray value here is
    # a hard model error rather than a soft one, so re-assert:
    #   frames  F = 8n+1  (temporal VAE compresses time 8x)
    #   dims    divisible by 64 (two-stage denoises at half res, then 2x's it)
    num_frames = config.snap_frames(num_frames)
    width, height = config.validate_dimensions(width, height)

    fitted_image = _fit_image(image_path, width, height)

    pipe = get_pipeline(variant)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    started = time.time()
    try:
        with torch.inference_mode():
            video, _audio, tiling_config = pipe(
                prompt=prompt,
                negative_prompt=config.NEGATIVE_PROMPT,
                seed=int(seed),
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=config.FPS,
                num_inference_steps=config.LOCAL_STEPS.get(variant, 8),
                video_guider_params=_guider_params(variant),
                # frame_idx=0 makes this the FIRST FRAME the clip grows from;
                # strength is Don's 0.62 (how tightly the render is pinned to it).
                # _fit_image does what fal used to do server-side, so ltx's own
                # centre-crop cannot silently eat the edges of the picture.
                images=[ImageConditioningInput(
                    path=str(fitted_image), frame_idx=0, strength=config.IMAGE_STRENGTH)],
                audio_path=str(audio_slice_path),
                audio_start_time=0.0,
                # The slice is already trimmed to exactly num_frames/fps by
                # jobs.py, so video and audio stay sample-locked in the mux.
                audio_max_duration=num_frames / config.FPS,
                stage_1_sigmas=_stage_1_sigmas(variant),
                tiling_config=AUTO_TILING,
            )

            # audio=None: write a SILENT mp4. The model's own audio is discarded
            # because mux.py overlays the original continuous track at the end.
            encode_video(
                video=video,
                fps=config.FPS,
                audio=None,
                output_path=str(out_path),
                video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
            )
    except Exception as e:
        apilog.local_render(
            variant=variant, width=width, height=height, num_frames=num_frames,
            ok=False, seconds=round(num_frames / config.FPS, 3),
            elapsed=round(time.time() - started, 2), job_id=job_id,
            error=f"{type(e).__name__}: {str(e)[:200]}")
        # A CUDA OOM leaves the allocator fragmented; give the next attempt
        # (usually a substitute image) a clean pool to work with.
        _empty_cache_on_oom(e)
        raise
    apilog.local_render(
        variant=variant, width=width, height=height, num_frames=num_frames,
        ok=True, seconds=round(num_frames / config.FPS, 3),
        elapsed=round(time.time() - started, 2), job_id=job_id)
    return out_path


def _empty_cache_on_oom(err: Exception) -> None:
    if "out of memory" not in str(err).lower():
        return
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def is_available() -> bool:
    """Cheap check the web layer uses for the 'render ready' badge.

    mock backend:  always ready (ffmpeg only).
    local backend: ready when CUDA is up, the LTX packages import, AND every
                   weight file is on disk - so a half-finished download shows
                   as not-ready instead of failing on the first Generate.
    """
    if config.RENDER_BACKEND == "mock":
        return True
    # Scoped to the DEFAULT variant: with the distilled checkpoint on disk the
    # app is genuinely usable, even while the dev checkpoint is still
    # downloading. Reporting "not ready" for an hour would be misleading.
    if config.missing_model_files(config.DEFAULT_VARIANT):
        return False
    try:
        torch, _ = _lazy_import()
        return bool(torch.cuda.is_available())
    except LtxNotInstalled:
        return False


def gpu_info() -> dict:
    """GPU name + VRAM for the status badge. Never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False, "name": None, "vram_gb": None}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda": True,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1024 ** 3, 1),
        }
    except Exception:
        return {"cuda": False, "name": None, "vram_gb": None}


# ---------------------------------------------------------------------------
# mock backend (ffmpeg-only, no model) - for local dev + pipeline testing
# ---------------------------------------------------------------------------
_MOCK_FONT_CACHE: "str | None | bool" = False   # False = not looked up yet


def _mock_font() -> Optional[str]:
    """Path to a usable TTF for ffmpeg's drawtext, escaped for a filtergraph.

    Returns None when no font is found, which makes the caller drop the text
    overlay rather than fail the render. Looked up once per process.
    """
    global _MOCK_FONT_CACHE
    if _MOCK_FONT_CACHE is not False:
        return _MOCK_FONT_CACHE  # type: ignore[return-value]

    candidates = [
        os.environ.get("SIZZLE_MOCK_FONT"),
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            # A Windows drive letter's ':' would be read as an option separator.
            # It has to survive TWO parsers - the filtergraph splitter and then
            # the option parser - so it needs two backslashes, not one:
            #   C:/Windows/Fonts/x.ttf  ->  C\\:/Windows/Fonts/x.ttf
            # (verified against ffmpeg 8.1; a single backslash fails to parse.)
            escaped = str(cand).replace("\\", "/").replace(":", r"\\:")
            _MOCK_FONT_CACHE = escaped
            return escaped
    _MOCK_FONT_CACHE = None
    return None

# Set via SIZZLE_MOCK_FAIL to a substring; any prompt containing it forces a
# render failure so the graceful black-fill / continue-resume path can be tested.
def _render_mock(
    *, image_path: Path, prompt: str, num_frames: int, out_path: Path,
    width: Optional[int] = None, height: Optional[int] = None,
) -> Path:
    """Synthesize a clip from the first-frame image with ffmpeg.

    The clip slow-zooms the image and burns in a live frame counter plus the
    first words of the prompt, with a hue that drifts over time. This makes the
    LAST frame of every clip visibly distinct, so continue-chains (which extract
    the last frame and feed it into the next clip) are verifiable by eye.
    """
    import subprocess

    fail_token = os.environ.get("SIZZLE_MOCK_FAIL")
    if fail_token and fail_token in prompt:
        raise RuntimeError(f"mock forced failure (SIZZLE_MOCK_FAIL='{fail_token}')")

    from . import config as _cfg
    w = width or _cfg.WIDTH
    h = height or _cfg.HEIGHT
    fps = _cfg.FPS
    n = max(1, int(num_frames))
    dur = n / fps
    label = (prompt or "sizzle").strip().split("\n")[0][:40].replace("'", "").replace(":", " ")

    # Fit the still into WxH (pad, keep aspect), slow zoom via scale over time,
    # drifting hue, and a burned-in frame counter + prompt label. drawtext uses
    # frame number n; hue rotates with t. All one filtergraph.
    vf_parts = [
        f"scale={w}:{h}:force_original_aspect_ratio=decrease",
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
        "hue=h=t*40",
    ]
    # drawtext needs a font. On Linux/macOS ffmpeg finds one through fontconfig;
    # Windows builds usually ship WITHOUT fontconfig, so an unqualified drawtext
    # dies with "Cannot load default config file" and takes the whole clip with
    # it. Name a real font file when we can find one, and simply drop the text
    # overlay when we cannot - a mock clip with no burned-in counter is still a
    # perfectly good mock clip.
    font = _mock_font()
    if font:
        vf_parts += [
            "drawbox=x=0:y=ih-88:w=iw:h=88:color=black@0.55:t=fill",
            f"drawtext=fontfile={font}:text='{label}':x=20:y=h-72:"
            f"fontsize=30:fontcolor=white@0.9",
            f"drawtext=fontfile={font}:text='MOCK f%{{n}}/{n}':x=20:y=h-38:"
            f"fontsize=26:fontcolor=0x3bffd0",
        ]
    vf = ",".join(vf_parts)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-t", f"{dur:.3f}",
        "-r", str(fps),
        "-frames:v", str(n),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"mock ffmpeg render failed:\n{proc.stderr[-2000:]}")
    return out_path
