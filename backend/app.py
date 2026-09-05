"""
Sizzle FastAPI app.

Serves the single-page timeline UI and the API behind it:

  POST /api/audio         upload music, get waveform + proposed segments
  POST /api/audio/manual  re-segment with explicit boundaries (manual markers)
  POST /api/image         upload one image, get an id + thumb url
  POST /api/autoprompt    given an image id, suggest a reactive prompt
  POST /api/generate      submit a timeline (segments bound to images+prompts)
  WS   /ws/{job_id}       live progress
  GET  /download/{job_id} the finished muxed mp4
  GET  /api/status        GPU/ffmpeg readiness badges

Everything renders on THIS machine's GPU: there is no fal.ai, no Cloudinary, and
no upload of your images or music anywhere. The UI and audio analysis work even
without the model weights present, so you can build a timeline while the ~100 GB
of checkpoints are still downloading — only Generate needs them.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import audio as audio_mod
from . import autoprompt
from . import config
from . import jobs
from . import ltx_engine
from . import mux

config.ensure_dirs()

app = FastAPI(title="Sizzle")

# in-memory session stores (ephemeral; die when you close the process)
_AUDIO: Dict[str, Path] = {}        # audio_id -> source path
_IMAGES: Dict[str, Path] = {}       # image_id -> image path

# ---------------------------------------------------------------------------
# Lightweight session siloing (no login — IMPROVEMENT-PLAN.md decision 7)
# ---------------------------------------------------------------------------
# Every first-time visitor is handed an httpOnly GUID cookie. Jobs are tagged
# with the owning session id; the WS stream, per-clip downloads and the finished
# mp4 check ownership so two people sharing the tunnel can't see or hijack each
# other's render. This is NOT auth — the session id is just the join point where
# real auth could slot in later. Render concurrency stays a GLOBAL single lock
# (jobs.is_busy), so only one render runs at a time regardless of who owns it.
_SID_COOKIE = "sizzle_sid"


def _new_sid() -> str:
    return uuid.uuid4().hex


@app.middleware("http")
async def _session_cookie(request: Request, call_next):
    sid = request.cookies.get(_SID_COOKIE)
    fresh = not sid
    if fresh:
        sid = _new_sid()
    request.state.sid = sid
    response = await call_next(request)
    if fresh:
        # httpOnly so page JS can't read it; lax so normal navigation carries it.
        response.set_cookie(
            _SID_COOKIE, sid, httponly=True, samesite="lax", path="/",
            max_age=60 * 60 * 24 * 30,
        )
    return response


@app.middleware("http")
async def _no_cache_html(request: Request, call_next):
    # Force HTML documents to always revalidate. The page links its css/js with
    # a ?v=N query so those stay cacheable and only reload when the version is
    # bumped -- but that only works if index.html itself isn't served stale.
    # Without this a browser can hand back a cached index.html that still points
    # at the old asset version, so edits appear to "not take" until a manual
    # hard-reload. no-cache here means the small HTML revalidates every load.
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


def _sid(request: Request) -> str:
    return getattr(request.state, "sid", None) or request.cookies.get(_SID_COOKIE) or ""


def _owns(job: "jobs.Job", sid: str) -> bool:
    """A job with no session (legacy / pre-cookie) is open to anyone; otherwise
    only its owning session may read its stream, clips or output."""
    return job.session_id is None or job.session_id == sid


@app.on_event("startup")
def _startup():
    jobs.start_worker()

    missing = config.missing_model_files()
    if config.RENDER_BACKEND == "local" and missing:
        print("\n[sizzle] local rendering is selected but weights are missing:")
        for label, path in missing:
            print(f"          - {label}: {path}")
        print("[sizzle] run:  python scripts/download_models.py")
        print("[sizzle] (the UI, timeline and audio analysis all work meanwhile;\n"
              "          set SIZZLE_BACKEND=mock to render placeholder clips)\n")

    # Optional: build the pipeline now instead of paying the model-load cost on
    # the first clip. Off by default so the server starts instantly; worth it on
    # a box that is going to render all day.
    if os.environ.get("SIZZLE_WARMUP") == "1" and not missing:
        def _warm():
            try:
                print("[sizzle] warming up the LTX-2.3 pipeline...")
                ltx_engine.warmup()
                print("[sizzle] pipeline warm.")
            except Exception as e:  # noqa: BLE001 - warmup must never block boot
                print(f"[sizzle] warmup failed ({type(e).__name__}: {e})")
        threading.Thread(target=_warm, name="sizzle-warmup", daemon=True).start()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def status(request: Request):
    import os
    sid = _sid(request)
    active = jobs.active_job()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    # Live reachability (not just key-presence) so a TLS-MITM / offline box shows
    # "unreachable" instead of a false green light (§6.2). None when no key.
    reach = autoprompt.probe_reachable() if has_key else {"reachable": None, "error": None}
    return {
        "gpu_ready": ltx_engine.is_available(),
        "backend": config.RENDER_BACKEND,
        "ffmpeg": mux.check_ffmpeg(),
        "autoprompt": has_key,
        "autoprompt_reachable": reach["reachable"],
        "autoprompt_error": reach["error"],
        "variants": {k: v["label"] for k, v in config.VARIANTS.items()},
        "default_variant": config.DEFAULT_VARIANT,
        "fps": config.FPS,
        "max_clip_seconds": config.MAX_CLIP_SECONDS,
        # format/resolution options: presets carry their frame megapixels so the
        # dropdown can hint at relative render TIME (local renders cost no money,
        # but time scales with pixels); default is the env WIDTH/HEIGHT.
        "formats": config.presets_payload(),
        "default_width": config.WIDTH,
        "default_height": config.HEIGHT,
        "min_dim": config.MIN_DIM,
        "max_dim": config.MAX_DIM,
        "dim_align": config.DIM_ALIGN,
        # Local-GPU status. `weights_missing` is what a fresh clone hits before
        # scripts/download_models.py has run, and it is the difference between
        # "no GPU" and "GPU fine, weights not downloaded yet" in the badge.
        "gpu": ltx_engine.gpu_info(),
        "weights_missing": [label for label, _ in config.missing_model_files()],
        # render-session state (Phase A/S): global single lock. `rendering` is
        # true whenever ANY visitor has a job in flight; `rendering_mine` says
        # whether it's this session's — so other clients can ghost GENERATE with
        # "someone is rendering" while the owner drives its own progress panel.
        "rendering": active is not None,
        "rendering_mine": bool(active and _owns(active, sid)),
    }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
@app.post("/api/audio")
async def upload_audio(file: UploadFile = File(...), mode: str = Form("beat")):
    audio_id = uuid.uuid4().hex[:12]
    dest = config.UPLOAD_DIR / f"{audio_id}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _AUDIO[audio_id] = dest

    analysis = audio_mod.analyze(dest, mode=mode)
    return {"audio_id": audio_id, **analysis.to_dict()}


@app.post("/api/audio/manual")
async def manual_segments(payload: dict):
    """
    Re-segment using explicit boundaries the user dropped in the UI.
    payload: {"audio_id": str, "boundaries": [t0, t1, t2, ...]}  seconds, sorted
    Boundaries define cut points; segments are consecutive pairs, each capped.
    """
    audio_id = payload["audio_id"]
    src = _AUDIO.get(audio_id)
    if not src:
        return JSONResponse({"error": "unknown audio_id"}, status_code=404)

    bounds = sorted(float(b) for b in payload["boundaries"])
    segs: List[dict] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        # split any over-cap span into <=MAX pieces
        t = start
        while t < end - 1e-3:
            e = min(t + config.MAX_CLIP_SECONDS, end)
            seg = audio_mod.Segment(len(segs), t, e)
            segs.append(seg.to_dict())
            t = e
    return {"audio_id": audio_id, "segments": segs}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
@app.post("/api/image")
async def upload_image(file: UploadFile = File(...)):
    image_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".png"
    dest = config.UPLOAD_DIR / f"img_{image_id}{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _IMAGES[image_id] = dest
    return {"image_id": image_id, "url": f"/api/image/{image_id}"}


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    p = _IMAGES.get(image_id)
    if not p or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p))


@app.post("/api/autoprompt")
async def auto_prompt(payload: dict):
    image_id = payload.get("image_id")
    p = _IMAGES.get(image_id)
    if not p:
        return JSONResponse({"error": "unknown image_id"}, status_code=404)
    res = autoprompt.suggest_detailed(p)
    # source == "heuristic" means the vision call didn't run (no key / connection
    # error); surfaced so the UI can warn instead of showing an identical prompt.
    # risk is the NSFW/platform-moderation hint (none|maybe|likely|unknown) from the
    # same vision call — drives the pre-flight "may be flagged" tray badge.
    return {"image_id": image_id, "prompt": res["prompt"],
            "source": res["source"], "error": res["error"],
            "risk": res.get("risk", "unknown")}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(payload: dict, request: Request):
    """
    payload: {
      "audio_id": str,
      "variant": "distilled"|"full",
      "seed": int|null,
      "blocks": [
        {"kind": "image", "start": float, "end": float,
         "image_id": str, "prompt": str},
        {"kind": "continue", "start": float, "end": float, "prompt": str},
        ...
      ]
    }

    Blocks are free-form: they need not cover the whole track. The render window
    is [min start, max end]; gaps inside it are auto-filled black. A "continue"
    block takes its first frame from the previous block's last frame.

    Back-compat: a legacy "clips" array (every entry an image) is also accepted.
    """
    # Global single-render lock (decision 3: NO queueing). A double-click or a
    # second person on the tunnel firing mid-render is rejected, not silently
    # queued — every extra job burns real fal credits and stomps the live log.
    if jobs.is_busy():
        return JSONResponse(
            {"error": "a render is already running — wait for it to finish"},
            status_code=409)

    audio_id = payload.get("audio_id")
    src = _AUDIO.get(audio_id)
    if not src:
        return JSONResponse({"error": "unknown audio_id"}, status_code=404)

    variant = payload.get("variant", config.DEFAULT_VARIANT)
    if variant not in config.VARIANTS:
        return JSONResponse({"error": "unknown variant"}, status_code=400)

    # Each 22B checkpoint is a separate ~46 GB download, so one variant can be
    # ready while the other is still coming down. Say so up front instead of
    # accepting the job and dying on the first clip with a stack trace.
    variant_missing = config.missing_model_files(variant)
    if config.RENDER_BACKEND == "local" and variant_missing:
        label = config.VARIANTS[variant]["label"]
        return JSONResponse(
            {"error": f"the '{label}' model is not downloaded yet "
                      f"({', '.join(l for l, _ in variant_missing)}). "
                      f"Pick another variant or finish the download."},
            status_code=409)

    # Optional per-job resolution (Phase E). Both must be present to override the
    # env default; validated permissively (bounds + even, not hard /32).
    width = payload.get("width")
    height = payload.get("height")
    if width is not None or height is not None:
        try:
            width, height = config.validate_dimensions(width, height)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
    else:
        width = height = None

    raw_blocks = payload.get("blocks")
    if raw_blocks is None:  # legacy: treat every clip as an image block
        raw_blocks = [{**c, "kind": "image"} for c in payload.get("clips", [])]

    def _num_frames(start: float, end: float) -> int:
        # Snap to a valid LTX frame count (F=8n+1); config clamps to
        # [MIN_FRAMES, MAX_FRAMES], so the MAX_CLIP_SECONDS ceiling is enforced
        # in frames and no block can ask the model for a count it will reject.
        return config.frames_for(max(0.0, end - start))

    blocks: List[jobs.Block] = []
    for i, b in enumerate(raw_blocks):
        kind = b.get("kind", "image")
        start, end = float(b["start"]), float(b["end"])
        if end <= start:
            return JSONResponse({"error": f"block {i} has non-positive length"}, status_code=400)
        prompt = (b.get("prompt") or "").strip()
        if kind == "image":
            img = _IMAGES.get(b.get("image_id"))
            if not img:
                return JSONResponse(
                    {"error": f"unknown image_id {b.get('image_id')}"}, status_code=400)
            blocks.append(jobs.Block(
                kind="image", index=i, start=start, end=end,
                num_frames=_num_frames(start, end), image_path=img,
                prompt=prompt or autoprompt._heuristic(img)))
        elif kind == "continue":
            blocks.append(jobs.Block(
                kind="continue", index=i, start=start, end=end,
                num_frames=_num_frames(start, end), image_path=None,
                prompt=prompt))
        else:
            return JSONResponse({"error": f"unknown block kind {kind!r}"}, status_code=400)

    if not blocks:
        return JSONResponse({"error": "no blocks"}, status_code=400)
    if all(b.kind == "continue" for b in blocks):
        return JSONResponse(
            {"error": "timeline needs at least one image block to seed the video"},
            status_code=400)

    job = jobs.submit(
        variant=variant, source_audio=src, blocks=blocks, seed=payload.get("seed"),
        session_id=_sid(request), width=width, height=height)
    return {"job_id": job.id, "total": job.total}


# ---------------------------------------------------------------------------
# Progress websocket
# ---------------------------------------------------------------------------
@app.websocket("/ws/{job_id}")
async def ws(websocket: WebSocket, job_id: str):
    # Ownership: cookies ride the WS handshake, so a second visitor can't attach
    # to someone else's job stream. Unknown/legacy jobs (no session) stay open.
    sid = websocket.cookies.get(_SID_COOKIE) or ""
    job = jobs.get_job(job_id)
    if job and not _owns(job, sid):
        await websocket.close(code=4403)
        return
    await websocket.accept()
    loop = asyncio.get_event_loop()
    q: "asyncio.Queue[dict]" = asyncio.Queue()

    # Reconnect support: the client sends how many events it already applied
    # (?since=N) so a socket dropped mid-render by a flaky IP catches back up
    # on exactly what it missed instead of replaying (and duplicating) the
    # whole log. The render itself is unaffected by the socket dropping — it
    # runs on the worker thread regardless of whether anyone is listening.
    try:
        since = int(websocket.query_params.get("since") or 0)
    except ValueError:
        since = 0

    def _cb(event: dict):
        loop.call_soon_threadsafe(q.put_nowait, event)

    jobs.register_listener(job_id, _cb, since=since)
    try:
        while True:
            event = await q.get()
            await websocket.send_text(json.dumps(event))
            if event.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        jobs.unregister_listener(job_id, _cb)


@app.get("/api/job/{job_id}")
def job_status(job_id: str, request: Request):
    """Plain-HTTP polling fallback for the progress WS. A network blip that
    stalls a WS upgrade often still lets a normal GET through, so the frontend
    polls this while its socket is down — enough to recover the download link
    (or the error) even if the live reconnect never manages to land."""
    job = jobs.get_job(job_id)
    if not job or not _owns(job, _sid(request)):
        return JSONResponse({"error": "not found"}, status_code=404)
    return jobs.snapshot(job)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
@app.get("/download/{job_id}")
def download(job_id: str, request: Request):
    job = jobs.get_job(job_id)
    if not job or not _owns(job, _sid(request)):
        return JSONResponse({"error": "not found"}, status_code=404)
    if not job.output_path or not job.output_path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return FileResponse(
        str(job.output_path),
        media_type="video/mp4",
        filename=f"sizzle_{job_id}.mp4",
    )


@app.get("/clip/{job_id}/{unit}")
def clip(job_id: str, unit: int, request: Request, dl: int = 0):
    """Serve one individual per-segment render so the progress log can link each
    clip as it finishes (spot-check before the final mux). Path is derived from
    ids only (no user string reaches the filesystem), and resolved inside
    CLIP_DIR to block traversal. `?dl=1` forces a download (Content-Disposition:
    attachment) instead of inline playback — decision 4, download-only links."""
    job = jobs.get_job(job_id)
    if not job or not _owns(job, _sid(request)):
        return JSONResponse({"error": "unknown job"}, status_code=404)
    path = (config.CLIP_DIR / f"{job_id}_u{int(unit)}.mp4").resolve()
    if config.CLIP_DIR.resolve() not in path.parents or not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    filename = f"{job_id}_clip{int(unit)}.mp4" if dl else None
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


# ---------------------------------------------------------------------------
# Static SPA (mounted last so /api/* wins)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")
