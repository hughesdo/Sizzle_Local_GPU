"""
Is this box ready to render? Exit 0 if yes, 1 if weights are missing.

The launchers call this so they can warn accurately. Checking for one
checkpoint by hand is not enough - a render needs the checkpoint AND the
audio-reactive LoRA AND the spatial upscaler AND the distilled LoRA AND the
Gemma text encoder, and missing any one of them fails at Generate time rather
than at startup.

    python scripts/check_weights.py          # human-readable
    python scripts/check_weights.py --quiet   # exit code only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config  # noqa: E402


def main() -> int:
    quiet = "--quiet" in sys.argv
    missing = config.missing_model_files()

    if config.RENDER_BACKEND == "mock":
        if not quiet:
            print("  backend is 'mock' - ffmpeg synthesizes clips, no weights needed.")
        return 0

    if not missing:
        if not quiet:
            print("  all model weights present - ready to render.")
        return 0

    if not quiet:
        print()
        print("  Not ready to render yet. Missing:")
        for label, path in missing:
            print(f"     - {label}")
        # The distilled path is what the default variant uses; you can render
        # with it even while the 'full quality' checkpoint is still downloading.
        needed_for_distilled = [l for l, _ in missing if l != "full checkpoint"]
        if not needed_for_distilled:
            print()
            print("  ...but that is ONLY the 'Full quality' variant. The default")
            print("  'Distilled (8-step, fast)' variant is ready to render right now.")
            return 0
        print()
        print("  Run:  .venv\\Scripts\\python.exe scripts\\download_models.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
