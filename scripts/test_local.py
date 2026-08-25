"""
Smoke-test the LOCAL GPU render path end to end, without the web server.

This is the replacement for the old scripts/test_fal.py: instead of spending fal
credits to prove the hosted path worked, it drives backend.ltx_engine directly
and proves the model on THIS box can turn an image + an audio slice into a clip.

    python scripts/test_local.py                     # synthetic image + beat track
    python scripts/test_local.py --image my.jpg --audio song.wav
    python scripts/test_local.py --seconds 3 --width 768 --height 448
    python scripts/test_local.py --variant full

It reports the wall-clock, the realtime factor, and peak VRAM, then ffprobes the
output so a clip that decodes to the wrong number of frames fails loudly rather
than looking fine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend import ltx_engine  # noqa: E402

OUT_DIR = ROOT / "_work" / "smoketest"


def make_test_image(path: Path, width: int, height: int) -> Path:
    """A structured, high-contrast still. The LoRA card asks for 'clear shapes,
    depth, light sources, geometry' - a flat gradient gives it nothing to move."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (8, 10, 18))
    d = ImageDraw.Draw(img)
    for i in range(28):  # concentric neon rings = obvious structure to deform
        t = i / 28
        box = [width * 0.5 - t * width * 0.7, height * 0.5 - t * height * 0.7,
               width * 0.5 + t * width * 0.7, height * 0.5 + t * height * 0.7]
        d.ellipse(box, outline=(int(20 + 200 * t), int(255 - 180 * t), int(160 + 60 * t)), width=3)
    for i in range(9):  # cubes along the bottom third
        x = width * (i + 0.5) / 9
        s = min(width, height) * 0.06
        d.rectangle([x - s, height * 0.72 - s, x + s, height * 0.72 + s],
                    outline=(240, 240, 255), width=2)
    img.save(path)
    return path


def make_test_audio(path: Path, seconds: float, sr: int = 48000) -> Path:
    """A hard 120 BPM kick pattern - unmistakable beats for the LoRA to lock to."""
    import numpy as np
    import soundfile as sf

    n = int(seconds * sr)
    t = np.arange(n) / sr
    audio = np.zeros(n, dtype=np.float32)
    for beat in range(int(seconds * 2) + 1):        # 120 BPM = 2 beats/sec
        start = int(beat * 0.5 * sr)
        if start >= n:
            break
        dur = min(int(0.18 * sr), n - start)
        env = np.exp(-np.linspace(0, 9, dur))       # sharp decay
        sweep = np.linspace(160, 45, dur)           # kick pitch drop
        audio[start:start + dur] += (
            env * np.sin(2 * np.pi * sweep * np.arange(dur) / sr)).astype(np.float32)
    audio += 0.04 * np.sin(2 * np.pi * 220 * t).astype(np.float32)   # pad
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), np.stack([audio, audio], axis=1), sr)
    return path


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets,width,height,r_frame_rate",
         "-count_packets", "-of", "json", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr[-500:]}")
    return json.loads(out.stdout)["streams"][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, default=None, help="first-frame image")
    ap.add_argument("--audio", type=Path, default=None, help="audio to react to")
    ap.add_argument("--seconds", type=float, default=3.0, help="clip length (default 3)")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--variant", default=config.DEFAULT_VARIANT, choices=list(config.VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = config.validate_dimensions(args.width, args.height)
    num_frames = config.frames_for(args.seconds)
    real_seconds = config.seconds_for(num_frames)

    print(f"backend      : {config.RENDER_BACKEND}")
    print(f"gpu          : {ltx_engine.gpu_info()}")
    missing = config.missing_model_files(args.variant)
    if config.RENDER_BACKEND == "local" and missing:
        print("\nMISSING WEIGHTS:")
        for label, path in missing:
            print(f"  - {label}: {path}")
        print("\nRun: python scripts/download_models.py")
        return 1

    image = args.image or make_test_image(OUT_DIR / "test_frame.png", width, height)
    audio = args.audio or make_test_audio(OUT_DIR / "test_beat.wav", real_seconds)
    prompt = args.prompt or (
        "sound-driven video, audio-reactive motion, continuous visual flow. "
        "Concentric neon rings pulse outward on every kick while the chrome cubes "
        "below slam, squash and rebound in time with the beat; the camera pushes "
        "slowly through the rings as light seams stretch and colour separates on "
        "each hit, deep cobalt and electric cyan against black glass."
    )

    print(f"variant      : {args.variant} ({config.VARIANTS[args.variant]['label']})")
    print(f"steps        : {config.LOCAL_STEPS.get(args.variant)}")
    print(f"quantization : {config.QUANTIZATION}")
    print(f"resolution   : {width}x{height}")
    print(f"frames       : {num_frames}  ({real_seconds:.2f}s @ {config.FPS}fps)")
    print(f"image        : {image}")
    print(f"audio        : {audio}")
    print("\nrendering (first run also loads ~46 GB of weights)...\n", flush=True)

    out = OUT_DIR / f"smoke_{args.variant}_{width}x{height}_{num_frames}f.mp4"
    started = time.time()
    ltx_engine.render_segment(
        variant=args.variant,
        image_path=Path(image),
        audio_slice_path=Path(audio),
        prompt=prompt,
        num_frames=num_frames,
        out_path=out,
        seed=args.seed,
        width=width,
        height=height,
        job_id="smoketest",
    )
    elapsed = time.time() - started

    try:
        import torch
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"peak VRAM    : {peak:.1f} GB")
    except Exception:
        pass

    print(f"elapsed      : {elapsed:.1f}s for {real_seconds:.2f}s of video "
          f"({elapsed / real_seconds:.1f}x realtime)")

    probe = ffprobe(out)
    got = int(probe["nb_read_packets"])
    print(f"output       : {out}")
    print(f"probed       : {probe['width']}x{probe['height']}, {got} frames @ {probe['r_frame_rate']}")

    ok = True
    if (probe["width"], probe["height"]) != (width, height):
        print(f"FAIL: expected {width}x{height}")
        ok = False
    # Allow a 1-frame slack: container packet counts can differ by one from the
    # encoded frame count depending on how the muxer flushes.
    if abs(got - num_frames) > 1:
        print(f"FAIL: expected {num_frames} frames, got {got}")
        ok = False

    print("\nPASS - local GPU rendering works." if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
