"""
Single-GPU job queue.

One worker thread, one FIFO. A job is a timeline: an ordered list of BLOCKS the
user placed. Each block is either

  - "image"    : a first-frame image + prompt, rendered for its duration, or
  - "continue" : no image of its own — its first frame is the LAST frame of the
                 previous rendered block, so the video flows straight on. Chains
                 of continues are allowed (continue after continue), which is how
                 a whole video can grow from a single seed image.

The timeline may be PARTIAL: blocks need not cover the whole track. The worker
renders only the window [firstBlock.start, lastBlock.end]; any uncovered gap
INSIDE that window becomes a black filler of the exact length so the audio
overlay never desyncs. The final mux overlays the ORIGINAL audio sliced to that
same window (Don's choice: music never breaks at a seam).

Graceful failure (Don's rule — never leave the user without their video):
  - image block fails  -> try substituting another image (bad input, OOM, etc.);
                          if all fail, black filler of the exact length.
  - continue fails     -> black filler of the exact length; the chain RESUMES on
                          the next block from the last good frame (not the black).
"""
from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import audio as audio_mod
from . import autoprompt
from . import config
from . import ltx_engine
from . import mux


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Block:
    kind: str                    # "image" | "continue"
    index: int
    start: float
    end: float
    num_frames: int
    prompt: str
    image_path: Optional[Path] = None   # image blocks only


@dataclass
class Job:
    id: str
    variant: str
    source_audio: Path
    blocks: List[Block]
    seed: Optional[int] = None
    session_id: Optional[str] = None  # owning browser session (GUID cookie); None = legacy
    width: Optional[int] = None        # render resolution for this job (None = config default)
    height: Optional[int] = None
    # runtime state
    status: str = "queued"           # queued|running|done|error
    current: int = 0
    total: int = 0
    window_start: float = 0.0
    window_end: float = 0.0
    output_path: Optional[Path] = None
    error: Optional[str] = None
    events: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure helpers
# ---------------------------------------------------------------------------
def _failure_reason(err: Exception) -> str:
    """Label a render failure for the live progress log.

    Rendering locally, there is no provider content filter left to trip - the
    realistic failures are the GPU running out of memory on an over-large
    request, or an input image the loader cannot decode. Naming them tells the
    user whether to drop the resolution or swap the image."""
    s = str(err).lower()
    if "out of memory" in s or "cuda oom" in s:
        return "GPU out of memory"
    if "missing" in s and "weight" in s:
        return "model weights missing"
    return "render error"


def _image_blocks(job: "Job") -> List[Block]:
    return [b for b in job.blocks if b.kind == "image" and b.image_path]


def _prompt_for_image(job: "Job", image_path: Path) -> str:
    """Prompt to use when substituting image_path: prefer a prompt already
    written for that image elsewhere in the timeline, else a neutral heuristic."""
    for b in _image_blocks(job):
        if b.image_path == image_path and b.prompt:
            return b.prompt
    return autoprompt._heuristic(image_path)


def _render_image_resilient(
    job: "Job", block: "Block", slice_path: Path, out_path: Path,
    done_images: List[Path],
) -> tuple[bool, "Path | None", "str | None"]:
    """Render an image block, substituting another image if the render fails.

    Attempt order: the block's own image, then images that already rendered OK
    this job, then any other image in the timeline. Bounded by
    RENDER_MAX_ATTEMPTS. Returns (ok, used_image, note)."""
    candidates: List[Path] = [block.image_path]  # type: ignore[list-item]
    for p in done_images:
        if p not in candidates:
            candidates.append(p)
    for b in _image_blocks(job):
        if b.image_path not in candidates:
            candidates.append(b.image_path)

    last_err: "Exception | None" = None
    attempts = 0
    for img in candidates:
        if attempts >= config.RENDER_MAX_ATTEMPTS:
            break
        is_original = img == block.image_path
        prompt = block.prompt if is_original else _prompt_for_image(job, img)
        try:
            attempts += 1
            ltx_engine.render_segment(
                variant=job.variant,
                image_path=img,
                audio_slice_path=slice_path,
                prompt=prompt,
                num_frames=block.num_frames,
                out_path=out_path,
                seed=job.seed,
                width=job.width,
                height=job.height,
                job_id=job.id,
            )
            note = None if is_original else f"original image failed; substituted {img.name}"
            return True, img, note
        except Exception as e:  # noqa: BLE001 - report and try next image
            last_err = e
            reason = _failure_reason(e)
            _emit(job, {"type": "clip_retry", "index": block.index,
                        "failed_image": img.name, "reason": reason,
                        "message": str(e)[:300]})
    return False, None, str(last_err) if last_err else "unknown render failure"


# ---------------------------------------------------------------------------
# Listener plumbing
# ---------------------------------------------------------------------------
_LISTENERS: Dict[str, List[Callable[[dict], None]]] = {}
_JOBS: Dict[str, Job] = {}
_Q: "queue.Queue[str]" = queue.Queue()
_LOCK = threading.Lock()


def _emit(job: Job, event: dict) -> None:
    event = {"job_id": job.id, **event}
    job.events.append(event)
    for cb in _LISTENERS.get(job.id, []):
        try:
            cb(event)
        except Exception:
            pass


def register_listener(job_id: str, cb: Callable[[dict], None]) -> None:
    _LISTENERS.setdefault(job_id, []).append(cb)
    job = _JOBS.get(job_id)
    if job:
        for e in job.events:
            try:
                cb(e)
            except Exception:
                pass


def unregister_listener(job_id: str, cb: Callable[[dict], None]) -> None:
    if job_id in _LISTENERS and cb in _LISTENERS[job_id]:
        _LISTENERS[job_id].remove(cb)


def is_busy() -> bool:
    """True if a render is queued or running. The app is a single-worker FIFO on
    one GPU, so this is a GLOBAL lock: only one render at a time across every
    visitor. With a 22B model resident in VRAM that is not just policy - two
    concurrent renders would fight over the card and OOM."""
    with _LOCK:
        return any(j.status in ("queued", "running") for j in _JOBS.values())


def active_job() -> Optional[Job]:
    """The currently queued/running job, if any (for ownership / status flags)."""
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return j
    return None


def submit(
    *, variant: str, source_audio: Path, blocks: List[Block],
    seed: Optional[int] = None, session_id: Optional[str] = None,
    width: Optional[int] = None, height: Optional[int] = None,
) -> Job:
    blocks = sorted(blocks, key=lambda b: b.start)
    job = Job(
        id=uuid.uuid4().hex[:12],
        variant=variant,
        source_audio=source_audio,
        blocks=blocks,
        seed=seed,
        session_id=session_id,
        width=width,
        height=height,
        total=len(blocks),
        window_start=blocks[0].start if blocks else 0.0,
        window_end=blocks[-1].end if blocks else 0.0,
    )
    with _LOCK:
        _JOBS[job.id] = job
    ahead = sum(1 for j in _JOBS.values()
                if j.status in ("queued", "running") and j.id != job.id)
    _emit(job, {"type": "queued", "position": ahead, "total": job.total})
    _Q.put(job.id)
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)


# ---------------------------------------------------------------------------
# Render units: tile the window with block renders + black gap fillers
# ---------------------------------------------------------------------------
@dataclass
class _Unit:
    kind: str          # "block" | "gap"
    start: float
    end: float
    num_frames: int
    block: Optional[Block] = None


def _build_units(job: Job) -> List[_Unit]:
    """Tile [window_start, window_end] with block renders and black gap fillers
    so the concatenated video is exactly the window length (audio stays synced)."""
    units: List[_Unit] = []
    cursor = job.window_start
    for b in job.blocks:
        if b.start - cursor > 0.05:  # uncovered gap before this block -> black
            gap_frames = max(1, round((b.start - cursor) * config.FPS))
            units.append(_Unit("gap", cursor, b.start, gap_frames))
        units.append(_Unit("block", b.start, b.end, b.num_frames, b))
        cursor = b.end
    return units


def _run_job(job: Job) -> None:
    job.status = "running"
    _emit(job, {"type": "start", "total": job.total,
                "window": [round(job.window_start, 3), round(job.window_end, 3)]})

    units = _build_units(job)
    # Total video length is fully determined by frame counts (num_frames is
    # authoritative on the render call), so we can slice audio to match exactly.
    total_frames = sum(u.num_frames for u in units)
    fps = config.FPS
    emitted_frames = 0                    # frames laid down so far in the window

    clip_paths: List[Path] = []
    done_images: List[Path] = []          # images that rendered OK (substitutes)
    last_good_frame: Optional[Path] = None  # for continue chains
    block_n = 0

    for u in units:
        unit_idx = len(clip_paths)          # stable id for this clip's file/link
        clip_out = config.CLIP_DIR / f"{job.id}_u{unit_idx}.mp4"
        clip_url = f"/clip/{job.id}/{unit_idx}"

        # The audio window for THIS unit is derived from the running frame cursor,
        # not the block's beat times: seg_start .. seg_start + num_frames/fps. This
        # is exactly the slice of the final overlay that plays over this clip, so
        # the reactivity we generate here lines up with what the viewer hears, and
        # clip length == its audio length to the frame (no cumulative drift).
        seg_start = job.window_start + emitted_frames / fps
        seg_end = seg_start + u.num_frames / fps
        emitted_frames += u.num_frames

        # ---- black gap filler (uncovered audio inside the window) ----
        if u.kind == "gap":
            mux.make_filler(clip_out, num_frames=u.num_frames, source_clip=None,
                            width=job.width, height=job.height)
            clip_paths.append(clip_out)
            _emit(job, {"type": "gap_filled", "seconds": round(u.end - u.start, 2),
                        "clip_url": clip_url})
            continue

        block = u.block  # type: ignore[assignment]
        block_n += 1
        job.current = block_n
        _emit(job, {"type": "clip_start", "index": block.index, "kind": block.kind,
                    "current": block_n, "total": job.total, "prompt": block.prompt})

        # slice this block's audio out of the source track (drives reactivity).
        # Length is exactly num_frames/fps (seg_start..seg_end above), matching
        # the clip the model will return frame-for-frame.
        slice_path = config.CLIP_DIR / f"{job.id}_seg{block.index}.wav"
        audio_mod.slice_audio(job.source_audio, seg_start, seg_end, slice_path)

        if block.kind == "image":
            ok, used_image, note = _render_image_resilient(
                job, block, slice_path, clip_out, done_images)
            if ok:
                if used_image is not None and used_image not in done_images:
                    done_images.append(used_image)
                clip_paths.append(clip_out)
                last_good_frame = _grab_last_frame(job, clip_out, last_good_frame)
                _emit(job, {"type": "clip_done", "index": block.index,
                            "current": block_n, "total": job.total,
                            "substituted": bool(note), "note": note,
                            "clip_url": clip_url})
            else:
                filler_src = clip_paths[-1] if clip_paths else None
                mux.make_filler(clip_out, num_frames=block.num_frames,
                                source_clip=filler_src,
                                width=job.width, height=job.height)
                clip_paths.append(clip_out)
                _emit(job, {"type": "clip_filled", "index": block.index,
                            "current": block_n, "total": job.total, "message": note,
                            "clip_url": clip_url})
            continue

        # ---- continue block: first frame = last good frame ----
        if last_good_frame is None or not last_good_frame.exists():
            # nothing to continue from (continue placed first, or all prior failed)
            mux.make_filler(clip_out, num_frames=block.num_frames, source_clip=None,
                            width=job.width, height=job.height)
            clip_paths.append(clip_out)
            _emit(job, {"type": "clip_filled", "index": block.index,
                        "current": block_n, "total": job.total,
                        "message": "no previous frame to continue from",
                        "clip_url": clip_url})
            continue

        try:
            ltx_engine.render_segment(
                variant=job.variant,
                image_path=last_good_frame,
                audio_slice_path=slice_path,
                prompt=block.prompt or "continue the motion smoothly from the previous frame",
                num_frames=block.num_frames,
                out_path=clip_out,
                seed=job.seed,
                width=job.width,
                height=job.height,
                job_id=job.id,
            )
            clip_paths.append(clip_out)
            last_good_frame = _grab_last_frame(job, clip_out, last_good_frame)
            _emit(job, {"type": "clip_done", "index": block.index, "kind": "continue",
                        "current": block_n, "total": job.total,
                        "substituted": False, "note": None, "clip_url": clip_url})
        except Exception as e:  # noqa: BLE001 - graceful black fill, resume chain
            reason = _failure_reason(e)
            _emit(job, {"type": "clip_retry", "index": block.index,
                        "failed_image": "continue", "reason": reason,
                        "message": str(e)[:300]})
            mux.make_filler(clip_out, num_frames=block.num_frames, source_clip=None,
                            width=job.width, height=job.height)
            clip_paths.append(clip_out)
            # last_good_frame UNCHANGED: next continue resumes from last good frame
            _emit(job, {"type": "clip_filled", "index": block.index,
                        "current": block_n, "total": job.total,
                        "message": f"continue failed ({reason}); black fill, chain resumes",
                        "clip_url": clip_url})

    # ---- assemble: window-audio slice + concat, X-safe mux ----
    # Trim the continuous overlay to the EXACT summed video length
    # (total_frames/fps), not the block-derived window_end. Frame snapping can
    # nudge the total a few frames off the raw window, and overlaying an audio
    # track a hair longer/shorter than the video is exactly what desyncs the
    # tail — this keeps them sample-locked.
    _emit(job, {"type": "muxing"})
    window_audio = config.CLIP_DIR / f"{job.id}_window.wav"
    audio_end = job.window_start + total_frames / fps
    audio_mod.slice_audio(job.source_audio, job.window_start, audio_end, window_audio)

    final = config.OUTPUT_DIR / f"{job.id}.mp4"
    mux.concat_and_mux(clip_paths=clip_paths, source_audio=window_audio, out_path=final)
    job.output_path = final
    job.status = "done"
    _emit(job, {"type": "done", "download": f"/download/{job.id}"})


def _grab_last_frame(job: Job, clip_path: Path, prev: Optional[Path]) -> Optional[Path]:
    """Extract clip's last frame for the next continue; keep prev on failure.
    Unique filename per clip so a failed extraction can't alias a good frame."""
    try:
        png = config.CLIP_DIR / f"{job.id}_lastframe_{clip_path.stem}.png"
        return mux.extract_last_frame(clip_path, png)
    except Exception as e:  # noqa: BLE001
        _emit(job, {"type": "clip_retry", "index": -1, "failed_image": "last-frame",
                    "reason": "frame-extract", "message": str(e)[:200]})
        return prev


def _worker() -> None:
    while True:
        job_id = _Q.get()
        job = _JOBS.get(job_id)
        if not job:
            continue
        try:
            _run_job(job)
        except Exception as e:
            job.status = "error"
            job.error = f"{e}\n{traceback.format_exc()}"
            _emit(job, {"type": "error", "message": str(e)})
        finally:
            _Q.task_done()


def start_worker() -> None:
    t = threading.Thread(target=_worker, name="sizzle-worker", daemon=True)
    t.start()
