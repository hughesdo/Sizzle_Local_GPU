"""
Admin-only call ledger.

Records the one remaining PAID outbound call - the Anthropic vision auto-prompt -
plus every LOCAL GPU render, to a JSONL file so Don can audit spend and see where
the wall-clock actually goes. This is an ADMIN file: it is never served to users
and never surfaced in the UI. One JSON object per line; tail/grep/jq friendly.

Local renders cost no money, so their rows carry `elapsed` (seconds of GPU time)
instead of a dollar estimate - that is the number worth watching now. There are
no fal or Cloudinary rows any more; this build makes neither call.

Deliberately best-effort: a logging failure must NEVER break a render, so every
write is wrapped and swallowed. Thread-safe (the render worker is a background
thread and the web layer is async).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from . import config

_LOCK = threading.Lock()


def _write(record: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **record}
    try:
        line = json.dumps(record, ensure_ascii=False)
        config.API_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with open(config.API_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass  # logging must never break a render


def anthropic_call(*, model, ok, input_tokens=None, output_tokens=None,
                   image=None, request_id=None, error=None) -> None:
    """Log one Anthropic vision (auto-prompt) call, with an estimated $ cost."""
    _write({
        "provider": "anthropic",
        "purpose": "autoprompt",
        "model": model,
        "ok": ok,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_cost_usd": config.anthropic_cost(model, input_tokens, output_tokens),
        "image": image,
        "request_id": request_id,
        "error": error,
    })


def local_render(*, variant, width, height, num_frames, ok,
                 seconds=None, elapsed=None, job_id=None, error=None) -> None:
    """Log one local GPU render segment.

    `seconds` is the length of video produced; `elapsed` is how long the GPU took
    to produce it. The ratio of the two is the throughput number to watch when
    tuning resolution, steps, or quantization.
    """
    _write({
        "provider": "local",
        "purpose": "render",
        "backend": config.RENDER_BACKEND,
        "variant": variant,
        "quantization": config.QUANTIZATION,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "seconds": seconds,
        "elapsed_sec": elapsed,
        "realtime_factor": (round(elapsed / seconds, 2)
                            if elapsed and seconds else None),
        "est_cost_usd": 0.0,
        "ok": ok,
        "job_id": job_id,
        "error": error,
    })
