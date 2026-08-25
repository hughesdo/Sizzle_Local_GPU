"""
Auto-prompt: look at a dropped image and suggest an LTX audio-reactive prompt
that fits what's in it.

Design: this calls the Anthropic API with the image so a vision model writes the
prompt. It is OPTIONAL, and it is now the ONLY outbound call the app can make -
rendering is entirely local. If no ANTHROPIC_API_KEY is set, the endpoint
degrades to a small heuristic stub so the app still works fully offline.

The prompt style follows LTX guidance: a single flowing paragraph, chronological,
literal, cinematographer's shot-list voice, under ~200 words. For performer
images it leans into singing/playing motion; for abstract/fractal it leans into
camera motion, morphing, color pulse, beat-synced geometry.
"""
from __future__ import annotations

import base64
import importlib
import os
import sys
from pathlib import Path

# Vision model that writes the per-image prompt. Override with
# SIZZLE_AUTOPROMPT_MODEL (e.g. claude-haiku-4-5 for cheaper/faster suggestions).
_MODEL = os.environ.get("SIZZLE_AUTOPROMPT_MODEL", "claude-opus-5")

_SYSTEM = (
    "You write first-frame prompts for the LTX-2.3 video model running an "
    "audio-reactive LoRA: the still image you are shown becomes frame one of "
    "a short clip whose motion is driven by a slice of music, so the visuals "
    "must PULSE, morph, and move in time with a beat.\n\n"
    "Look closely at THIS specific image and write a prompt grounded in what is "
    "actually in it - the real subject, setting, palette, textures, and lighting "
    "- not a generic template. Two different images must yield two clearly "
    "different prompts. Then describe motion that would ride the music:\n"
    "- A person/performer: singing or playing motion, breath, expression, hair "
    "and fabric movement, stage or ambient light pulsing on the beat.\n"
    "- A face/portrait: subtle head motion, eyes, micro-expressions, light and "
    "color throbbing with the rhythm.\n"
    "- Abstract / fractal / texture: camera push or pull, morphing geometry, "
    "liquid color pulses, beat-synced shimmer and bloom.\n"
    "- A scene/landscape/object: a slow camera move plus reactive light, "
    "particles, and rippling detail keyed to the beat.\n\n"
    "Style: ONE flowing paragraph, chronological and literal, in a "
    "cinematographer's shot-list voice, under 120 words. Start directly with the "
    "action. No preamble, no quotes, no title, no mention of audio files or "
    "LoRAs.\n\n"
    "After the prompt paragraph, add a final line by itself in EXACTLY this form:\n"
    "  RISK: none|maybe|likely\n"
    "This rates how likely a video built from THIS image is to be age-gated or "
    "taken down by automated moderation on the platforms these clips get posted "
    "to - YouTube, TikTok, Instagram (nudity, explicit sexual content, or very "
    "revealing/suggestive clothing like lingerie or micro skirts tend to trip "
    "it; ordinary photos do not). 'none' = clearly safe, 'maybe' = "
    "borderline/suggestive, 'likely' = probably flagged. Output the prompt "
    "paragraph, then that one RISK line, and nothing else."
)

_RISK_VALUES = {"none", "maybe", "likely"}


def _split_risk(text: str) -> tuple[str, str]:
    """Pull the trailing 'RISK: <value>' line off the model output. Returns
    (prompt_without_risk_line, risk). risk is 'unknown' if the model omitted or
    mangled the tag, so a parsing miss never fabricates a false safe/unsafe call.
    Keeps the audio-reactive prompt itself byte-identical minus the tag."""
    risk = "unknown"
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("risk:"):
            val = low.split(":", 1)[1].strip()
            if val in _RISK_VALUES:
                risk = val
            del lines[i]
        break  # only inspect the last non-empty line
    return "\n".join(lines).strip(), risk


def _http_client():
    """An httpx client that trusts the resolved CA bundle, built from the SAME
    httpx major version the installed Anthropic SDK expects.

    The SDK moved to httpx 2.x (distributed under the `httpx2` package name) in
    anthropic 1.0, and it type-checks this argument: handing it a plain
    `httpx.Client` raises

        TypeError: Invalid `http_client` argument;
                   Expected an instance of `httpx2.Client`

    which the caller's except-block turned into a silent fall back to the
    generic prompt - i.e. exactly the "all my prompts look the same" symptom
    the CA-bundle wiring was added to fix in the first place, with a different
    cause. Resolving the module off the SDK itself keeps this working across
    the 0.x/1.x split in either direction.

    Returns None when no bundle is configured or no usable client class is
    found; the SDK then builds its own client, which is the right default.
    """
    from . import config
    if not config.CA_BUNDLE:
        return None

    candidates = []
    try:  # whichever module the installed SDK actually imported
        base = importlib.import_module("anthropic._base_client")
        candidates += [m for m in (getattr(base, "httpx2", None),
                                   getattr(base, "httpx", None)) if m is not None]
    except Exception:
        pass
    for name in ("httpx2", "httpx"):      # then plain preference order
        try:
            candidates.append(importlib.import_module(name))
        except ImportError:
            continue

    for mod in candidates:
        try:
            return mod.Client(verify=config.CA_BUNDLE)
        except Exception:
            continue
    return None


def _b64(image_path: Path) -> tuple[str, str]:
    data = image_path.read_bytes()
    ext = image_path.suffix.lower().lstrip(".")
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")
    return base64.b64encode(data).decode(), media


def _heuristic(image_path: Path) -> str:
    """Offline fallback: generic reactive prompt, no vision."""
    return (
        "The image comes alive with subtle motion; the camera drifts slowly "
        "inward as light pulses in time with the music, colors shifting and "
        "shimmering across the frame, fine details rippling and reacting to "
        "each beat, the whole scene breathing with the rhythm."
    )


def suggest_detailed(image_path: Path) -> dict:
    """Suggest a prompt and report HOW it was produced.

    Returns {"prompt": str, "source": "vision"|"heuristic", "error": str|None,
             "risk": "none"|"maybe"|"likely"|"unknown"}.
    'source' lets the caller tell the difference between a real vision prompt and
    the offline fallback - without it, an SSL/connection failure silently returns
    the same heuristic string for every image and looks like the model is just
    boring (this was the "all my prompts are identical" bug). 'risk' is a
    NSFW / platform-moderation hint from the SAME vision call (a few extra output
    tokens, no second request); it drives the pre-flight "may be flagged" badge.
    Any non-vision path returns risk 'unknown' so no badge is shown.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"prompt": _heuristic(image_path), "source": "heuristic",
                "error": "no ANTHROPIC_API_KEY", "risk": "unknown"}
    try:
        import anthropic
        # Verify TLS against the resolved CA bundle (includes the Avast MITM root
        # on this box); the SDK otherwise defaults to certifi and fails behind the
        # antivirus proxy with CERTIFICATE_VERIFY_FAILED.
        try:
            client = anthropic.Anthropic(api_key=api_key, http_client=_http_client())
        except TypeError:
            # A future SDK could tighten this check again. A working auto-prompt
            # on the default trust store beats no auto-prompt at all, so drop the
            # custom client rather than losing the call. (config.py also exports
            # SSL_CERT_FILE, which most stacks honour on their own.)
            client = anthropic.Anthropic(api_key=api_key)
        b64, media = _b64(image_path)
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            # low effort: this is short creative text, not a reasoning task -
            # keeps latency and cost down and leaves room under max_tokens.
            output_config={"effort": "low"},
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media, "data": b64}},
                    {"type": "text", "text":
                        "Write the reactive first-frame prompt for THIS image."},
                ],
            }],
        )
        # admin cost ledger: one paid vision call (see backend/apilog.py)
        _log_call(msg, image_path, ok=True)
        parts = [b.text for b in msg.content if getattr(b, "type", "") == "text"]
        out = "\n".join(parts).strip()
        if out:
            prompt, risk = _split_risk(out)
            return {"prompt": prompt or out, "source": "vision", "error": None,
                    "risk": risk}
        return {"prompt": _heuristic(image_path), "source": "heuristic",
                "error": "empty vision response", "risk": "unknown"}
    except Exception as e:
        # Don't fail the request, but make the reason visible in the server log
        # AND in the response so the UI can warn instead of silently degrading.
        print(f"[autoprompt] vision call failed ({type(e).__name__}: {e}); "
              f"using heuristic fallback", file=sys.stderr)
        _log_call(None, image_path, ok=False, error=f"{type(e).__name__}: {e}")
        return {"prompt": _heuristic(image_path), "source": "heuristic",
                "error": f"{type(e).__name__}: {e}", "risk": "unknown"}


def _log_call(msg, image_path: Path, *, ok: bool, error: str | None = None) -> None:
    """Record the vision call in the admin API ledger (tokens + est cost)."""
    from . import apilog
    usage = getattr(msg, "usage", None)
    apilog.anthropic_call(
        model=_MODEL, ok=ok,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        image=image_path.name,
        request_id=getattr(msg, "_request_id", None),
        error=error,
    )


def suggest(image_path: Path) -> str:
    """Back-compat: just the prompt string."""
    return suggest_detailed(image_path)["prompt"]


# ---------------------------------------------------------------------------
# Reachability probe (IMPROVEMENT-PLAN.md §6.2 / Phase D4)
# ---------------------------------------------------------------------------
# /api/status used to report auto-prompt "on" from mere key-presence. During the
# Avast TLS-MITM outage the key WAS present but every call failed TLS verification,
# so the badge showed green while nothing worked. This does a cheap, cached TLS
# handshake to the API host through the SAME CA bundle the real client uses, so
# the badge reflects whether calls will actually succeed - not just whether a key
# exists. Cached so the frequent status poll doesn't reconnect every few seconds.
_PROBE = {"ok": None, "ts": 0.0, "error": None}


def probe_reachable(ttl: float = 120.0) -> dict:
    """Return {"reachable": True|False|None, "error": str|None}. None means "not
    applicable" (no key set — nothing to probe)."""
    import time
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"reachable": None, "error": "no key"}
    now = time.time()
    if _PROBE["ok"] is not None and now - _PROBE["ts"] < ttl:
        return {"reachable": _PROBE["ok"], "error": _PROBE["error"]}
    import socket
    import ssl
    from . import config
    ok, err = True, None
    try:
        ctx = ssl.create_default_context(cafile=config.CA_BUNDLE or None)
        with socket.create_connection(("api.anthropic.com", 443), timeout=4) as sock:
            with ctx.wrap_socket(sock, server_hostname="api.anthropic.com"):
                pass  # a successful TLS handshake is exactly what MITM breaks
    except Exception as e:  # noqa: BLE001 - any failure = treat as unreachable
        ok, err = False, f"{type(e).__name__}: {e}"
    _PROBE.update(ok=ok, ts=now, error=err)
    return {"reachable": ok, "error": err}
