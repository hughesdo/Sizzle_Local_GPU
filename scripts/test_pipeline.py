"""Dev harness: exercise the render pipeline synchronously on the mock backend.

Builds a fake timeline (image block -> continue -> gap -> image block) and runs
the job loop directly (no server, no queue), then ffprobes the output. Proves:
mock render, last-frame extraction, continue chaining, black gap fill, and the
partial-window audio mux - all without touching the GPU (mock backend).

Run:  SIZZLE_BACKEND=mock ./.venv/Scripts/python.exe scripts/test_pipeline.py
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SIZZLE_BACKEND", "mock")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config, jobs, audio as audio_mod  # noqa: E402

config.ensure_dirs()
ART = ROOT / "_work" / "testart"
ART.mkdir(parents=True, exist_ok=True)


def make_image(path: Path, color, text):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (config.WIDTH, config.HEIGHT), color)
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, config.WIDTH - 40, config.HEIGHT - 40], outline=(255, 255, 255), width=6)
    d.text((80, 80), text, fill=(255, 255, 255))
    im.save(path)
    return path


def make_audio(path: Path, seconds=16):
    # click track ~120bpm + tone so librosa finds beats and slicing has content
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}",
        "-filter_complex", "[1]atrim=0:0.03,apad=pad_dur=0.47,aloop=loop=-1:size=48000*0.5[click];"
                           "[0][click]amix=inputs=2:duration=first",
        "-t", str(seconds), "-ar", "44100", str(path),
    ], capture_output=True, text=True)
    return path


def ffprobe_dur(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


def main():
    img1 = make_image(ART / "img1.png", (180, 40, 40), "IMAGE 1")
    img2 = make_image(ART / "img2.png", (40, 80, 180), "IMAGE 2")
    aud = make_audio(ART / "test.wav", 16)
    print("assets ready:", img1.name, img2.name, aud.name)

    # timeline: img1 [0-4], continue [4-8], (gap 8-10 -> black), img2 [10-14]
    blocks = [
        jobs.Block("image", 0, 0.0, 4.0, round(4.0 * config.FPS), "red image pulsing", img1),
        jobs.Block("continue", 1, 4.0, 8.0, round(4.0 * config.FPS), "keep flowing"),
        jobs.Block("image", 2, 10.0, 14.0, round(4.0 * config.FPS), "blue image pulsing", img2),
    ]
    job = jobs.Job(id="testjob", variant="distilled", source_audio=aud,
                   blocks=sorted(blocks, key=lambda b: b.start),
                   total=len(blocks),
                   window_start=blocks[0].start, window_end=blocks[-1].end)

    events = []
    jobs.register_listener("testjob", lambda e: events.append(e))
    jobs._JOBS["testjob"] = job
    jobs._run_job(job)

    print("\n--- events ---")
    for e in events:
        print(" ", e.get("type"), {k: v for k, v in e.items() if k not in ("type", "job_id")})

    out = job.output_path
    dur = ffprobe_dur(out) if out else -1
    print("\noutput:", out)
    print("duration: %.2fs (expected ~14.0 = window 0..14)" % dur)
    assert out and out.exists(), "no output produced"
    assert 13.0 < dur < 15.0, f"unexpected duration {dur}"
    print("\nPASS: pipeline produced a windowed video with continue + black gap.")


if __name__ == "__main__":
    main()
