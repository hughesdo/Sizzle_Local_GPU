"""
End-to-end API smoke test for Sizzle (mock backend), driving the real HTTP + WS
surface the browser uses. Covers:

  T2  audio upload -> waveform peaks + beats + downbeats + tempo
  T3  autoprompt VARIES per image (the bug Don suspected) via real Anthropic call
  PIPE  /api/generate with image + gap(black-fill) + continue block, end to end,
        WS progress captured, output mp4 downloaded and ffprobe'd for duration

Run against an already-running server on 127.0.0.1:8000.
Usage: python scripts/test_api.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import websockets

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "_work" / "uploads"
# ffprobe from PATH. (This used to be an absolute path into one particular
# machine's ffmpeg build, which made the script unrunnable anywhere else.)
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# heuristic fallback string (must match autoprompt._heuristic) so we can prove
# the vision path produced something REAL, not the offline stub.
HEURISTIC_SNIPPET = "The image comes alive with subtle motion"

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}" + (f" :: {detail}" if detail else ""))
    return ok


def pick_audio() -> Path:
    for cand in ("2c0215d4581a_wasn't me.mp3",):
        p = UP / cand
        if p.exists():
            return p
    # fallback: any audio in uploads
    for p in UP.iterdir():
        if p.suffix.lower() in (".mp3", ".wav", ".flac"):
            return p
    raise SystemExit("no test audio found in _work/uploads")


def pick_images(n) -> list[Path]:
    imgs = sorted(p for p in UP.iterdir()
                  if p.name.startswith("img_") and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    # spread across the list so we get genuinely different pictures
    if len(imgs) < n:
        raise SystemExit("need more test images in _work/uploads")
    step = max(1, len(imgs) // n)
    return [imgs[i * step] for i in range(n)]


def upload_audio(path: Path) -> dict:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/audio", files={"file": (path.name, f)},
                          data={"mode": "beat"}, timeout=120)
    r.raise_for_status()
    return r.json()


def upload_image(path: Path) -> str:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/image", files={"file": (path.name, f)}, timeout=60)
    r.raise_for_status()
    return r.json()["image_id"]


def autoprompt(image_id: str) -> str:
    r = requests.post(f"{BASE}/api/autoprompt", json={"image_id": image_id}, timeout=120)
    r.raise_for_status()
    return r.json()["prompt"]


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)], text=True)
    return float(json.loads(out)["format"]["duration"])


async def run_generate(payload: dict) -> tuple[dict, list]:
    r = requests.post(f"{BASE}/api/generate", json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        return j, []
    job_id, total = j["job_id"], j["total"]
    events = []
    uri = f"ws://127.0.0.1:8000/ws/{job_id}"
    async with websockets.connect(uri) as ws:
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=180)
            except asyncio.TimeoutError:
                events.append({"type": "_timeout"})
                break
            e = json.loads(msg)
            events.append(e)
            if e.get("type") in ("done", "error"):
                break
    return j, events


def main():
    print("=== SIZZLE API E2E (mock backend) ===\n")

    st = requests.get(f"{BASE}/api/status", timeout=10).json()
    print(f"status: backend={st['backend']} ffmpeg={st['ffmpeg']} "
          f"autoprompt={st['autoprompt']} fps={st['fps']} max_clip={st['max_clip_seconds']}\n")
    check("status backend is mock", st["backend"] == "mock", st["backend"])

    # ---- T2: audio analysis -------------------------------------------------
    print("\n-- T2: audio upload + analysis --")
    a = upload_audio(pick_audio())
    dur = a.get("duration", 0)
    peaks = a.get("peaks", [])
    beats = a.get("beats", [])
    downs = a.get("downbeats", [])
    tempo = a.get("tempo", 0)
    print(f"  duration={dur:.1f}s tempo={tempo} peaks={len(peaks)} "
          f"beats={len(beats)} downbeats={len(downs)}")
    check("audio duration > 0", dur > 0, f"{dur:.1f}s")
    check("waveform peaks present", len(peaks) > 10, f"{len(peaks)} peaks")
    check("beats detected", len(beats) > 4, f"{len(beats)} beats")
    check("downbeats present", len(downs) > 0, f"{len(downs)} downbeats")
    check("beats within duration", all(0 <= b <= dur + 0.5 for b in beats),
          f"max beat {max(beats) if beats else 0:.1f} vs dur {dur:.1f}")
    audio_id = a["audio_id"]

    # ---- T3: autoprompt VARIES ---------------------------------------------
    print("\n-- T3: autoprompt varies per image (Don's suspected bug) --")
    imgs = pick_images(3)
    prompts = []
    for p in imgs:
        iid = upload_image(p)
        t0 = time.time()
        pr = autoprompt(iid)
        print(f"  {p.name}: ({time.time()-t0:.1f}s) {pr[:90]}...")
        prompts.append(pr)
    non_empty = all(len(p) > 20 for p in prompts)
    check("all prompts non-empty", non_empty)
    distinct = len(set(prompts)) == len(prompts)
    check("all prompts distinct from each other", distinct,
          f"{len(set(prompts))}/{len(prompts)} unique")
    real = all(HEURISTIC_SNIPPET not in p for p in prompts)
    check("prompts are vision-generated (not heuristic fallback)", real)

    # ---- PIPELINE: blocks + gap black-fill + continue ----------------------
    print("\n-- PIPELINE: image + gap(black-fill) + continue, end to end --")
    p_imgs = pick_images(2)
    id_a = upload_image(p_imgs[0])
    id_b = upload_image(p_imgs[1])
    # timeline: A[0-4], GAP[4-6], B[6-9], CONTINUE[9-13] (extends B)
    blocks = [
        {"kind": "image", "start": 0.0, "end": 4.0, "image_id": id_a, "prompt": "clip A"},
        {"kind": "image", "start": 6.0, "end": 9.0, "image_id": id_b, "prompt": "clip B"},
        {"kind": "continue", "start": 9.0, "end": 13.0, "prompt": "extend B"},
    ]
    window = (0.0, 13.0)
    payload = {"audio_id": audio_id, "variant": "distilled", "seed": 42, "blocks": blocks}
    job, events = asyncio.run(run_generate(payload))
    if "error" in job:
        check("generate accepted blocks", False, job["error"])
    else:
        types = [e["type"] for e in events]
        print(f"  job {job['job_id']} total={job['total']} events={types}")
        check("generate accepted blocks", True, f"total={job['total']}")
        check("got a gap_filled event (black fill)", "gap_filled" in types)
        check("job reached 'done'", types and types[-1] == "done",
              types[-1] if types else "no events")
        done = next((e for e in events if e["type"] == "done"), None)
        if done:
            dl = done.get("download")
            r = requests.get(f"{BASE}{dl}", timeout=60)
            out = ROOT / "_work" / "outputs" / f"apitest_{job['job_id']}.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r.content)
            vd = ffprobe_duration(out)
            print(f"  downloaded {out.name} ({len(r.content)} bytes) duration={vd:.2f}s")
            check("output mp4 downloaded", len(r.content) > 1000)
            check("output duration ~= render window (13s)", abs(vd - 13.0) < 1.0,
                  f"{vd:.2f}s vs 13s")

    # ---- summary ------------------------------------------------------------
    npass = sum(1 for _, ok in results if ok)
    print(f"\n=== {npass}/{len(results)} checks passed ===")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
