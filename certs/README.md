# certs/

Sizzle looks for a CA bundle here (`certs/ca-bundle.pem`) so its HTTPS calls
verify correctly **even on a machine whose antivirus does TLS inspection**
(e.g. Avast, or a corporate MITM proxy). Those
products re-sign HTTPS with their own root, which lives in the OS trust store
but *not* in Python's bundled `certifi` — so without this, every request fails
with `CERTIFICATE_VERIFY_FAILED`.

Rendering is entirely local now, so this only affects two things: the optional
Anthropic auto-prompt call, and `scripts/download_models.py` pulling weights
from Hugging Face. Once the weights are on disk, Sizzle renders with no network
at all.

**The bundle is intentionally not committed** — it's machine-specific. If it's
absent, Sizzle falls back to `certifi` automatically, which is correct on any
normal machine. You only need to generate one if you're behind a TLS-inspecting
proxy:

- **Windows:** run `scripts/fix-certs.ps1` (appends your OS trusted roots into
  certifi), or drop your combined bundle here as `ca-bundle.pem`.
- **Any OS:** point `SIZZLE_CA_BUNDLE=/path/to/your-bundle.pem` at it instead.

Resolution order (see `backend/config.py`): `$SIZZLE_CA_BUNDLE` → `certs/ca-bundle.pem` → `certifi`.
