"""
Final assembly with ffmpeg.

Strategy (Don's pick): render each segment's video independently, then lay the
ORIGINAL continuous audio over the concatenated video so the music never breaks
at a clip seam. The per-clip audio the model generated is discarded for the
final; it only ever drove the reactive motion during generation.

Encode with NVENC for speed (you're already on the GPU), libx264 fallback.
Audio muxed with Don's X-safe flags: aac 192k, 48k, +faststart.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

from . import config


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr[-4000:]}"
        )


def _video_encoder() -> List[str]:
    if config.USE_NVENC:
        # p5 preset ~ balanced; vbr for quality. Falls back if NVENC absent.
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p"]


def concat_and_mux(
    *, clip_paths: List[Path], source_audio: Path, out_path: Path
) -> Path:
    if not clip_paths:
        raise ValueError("no clips to concat")

    workdir = out_path.parent
    concat_list = workdir / f"{out_path.stem}_concat.txt"
    silent_concat = workdir / f"{out_path.stem}_video.mp4"

    # 1. concat the video-only stream. Re-encode (not stream-copy) so mismatched
    #    clip headers/timebases don't corrupt the join.
    with open(concat_list, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-an",                                   # drop per-clip audio here
        *_video_encoder(),
        "-r", str(config.FPS),
        str(silent_concat),
    ]
    try:
        _run(concat_cmd)
    except RuntimeError:
        # NVENC not available on this box -> retry once with libx264
        if config.USE_NVENC:
            config.USE_NVENC = False  # session fallback
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-an", *_video_encoder(), "-r", str(config.FPS),
                str(silent_concat),
            ]
            _run(concat_cmd)
        else:
            raise

    # 2. mux the continuous source audio onto the concatenated video.
    #    -shortest trims audio if it slightly overruns the summed video length.
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_concat),
        "-i", str(source_audio),
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        *config.AUDIO_ARGS,
        *config.FASTSTART_ARGS,
        "-shortest",
        str(out_path),
    ]
    _run(mux_cmd)

    # tidy intermediates
    for p in (concat_list, silent_concat):
        try:
            p.unlink()
        except OSError:
            pass
    return out_path


def make_filler(
    out_path: Path, *, num_frames: int, source_clip: Path | None = None,
    width: int | None = None, height: int | None = None,
) -> Path:
    """Produce a placeholder clip of exactly num_frames so a failed segment does
    not shorten the timeline or desync the audio overlay. If a previously
    rendered clip is given, loop it to the needed length (keeps the video
    visually consistent); otherwise emit a black clip of the same length.

    width/height MUST be the job's render size, not the config default: a
    portrait job whose segment failed would otherwise get a landscape black
    filler spliced into the middle of it, and the concat re-encode would fight
    over which size wins. None falls back to the config default.
    """
    w = width or config.WIDTH
    h = height or config.HEIGHT
    if source_clip is not None and source_clip.exists():
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(source_clip),
            "-frames:v", str(num_frames),
            "-an", *_video_encoder(), "-r", str(config.FPS),
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={w}x{h}:r={config.FPS}",
            "-frames:v", str(num_frames),
            "-an", *_video_encoder(),
            str(out_path),
        ]
    try:
        _run(cmd)
    except RuntimeError:
        if config.USE_NVENC:
            config.USE_NVENC = False  # NVENC missing -> retry once on libx264
            return make_filler(out_path, num_frames=num_frames,
                               source_clip=source_clip, width=w, height=h)
        raise
    return out_path


def extract_last_frame(clip_path: Path, out_png: Path) -> Path:
    """Grab the final frame of a rendered clip as a PNG.

    This is the heart of a "continue": the last frame of clip N becomes the
    first-frame conditioning image for clip N+1, so the video keeps flowing from
    exactly where the previous clip ended. `-update 1` + `-frames:v 1` while
    seeking near the end reliably lands the last decodable frame.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # sseof -N seeks N seconds before EOF; -3f is not portable so use a small
    # negative offset and take the last frame via the reverse+trim trick, which
    # is robust across containers.
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-1",           # last ~1s
        "-i", str(clip_path),
        "-vf", "reverse",         # first frame after reverse == original last
        "-frames:v", "1",
        "-update", "1",
        str(out_png),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_png.exists():
        # Robust fallback: decode the whole clip writing each frame over the same
        # file (-update 1); whatever survives is the final frame. Always works.
        cmd2 = ["ffmpeg", "-y", "-i", str(clip_path), "-update", "1", str(out_png)]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        if proc2.returncode != 0 or not out_png.exists():
            raise RuntimeError(
                f"could not extract last frame from {clip_path.name}:\n"
                f"{proc.stderr[-1500:]}\n---fallback---\n{proc2.stderr[-1500:]}"
            )
    return out_png


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None
