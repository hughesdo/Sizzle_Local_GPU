# `_work/` — runtime scratch

**Nothing in here is in git, and nothing in here needs backing up.**
`config.ensure_dirs()` recreates the whole tree at startup.

This file exists so the directory's purpose is documented even though its
contents never are.

## Layout

| Path | Created by | Contents |
|---|---|---|
| `uploads/` | `app.py` | Images and audio as users drop them in, namespaced per session |
| `clips/` | `jobs.py` | Per-segment rendered mp4s — **silent by design** (see below) |
| `outputs/` | `mux.py` | The final muxed mp4s users download |
| `logs/` | `apilog.py` | `api-calls.jsonl`, the admin ledger |
| `logs/api-calls.jsonl` | `apilog.py` | One JSON object per line: every Anthropic vision call (the only paid API) and every local GPU render, timed |

Also generated at the repo root by the launchers: `server.log`, `server.log.err`,
`tunnel.log` — all gitignored.

## Things worth knowing

**The per-clip mp4s in `clips/` have no audio, and that is correct.**
`ltx_engine.py` discards the model's generated audio; `mux.py` lays the
*original* continuous track over the concatenated timeline so the music never
breaks at a clip seam. A silent clip is not a failed render.

**Black filler clips are a feature.** If a segment fails — bad input image, CUDA
OOM — the worker substitutes another image, and if that fails too it writes
black of the *exact* length. The mux stays gapless and the audio never desyncs.
You'll see these in `clips/` after a failure.

**`api-calls.jsonl` is admin-only.** It is never served to users and never
surfaced in the UI. Local renders cost no money, so their rows carry `elapsed`
(seconds of GPU time) rather than a dollar estimate — that's the number worth
watching now. `jq`-friendly:

```bash
jq -r 'select(.kind=="render") | .elapsed' _work/logs/api-calls.jsonl
```

**It is safe to delete this whole directory** when the server is stopped. You
lose finished videos users haven't downloaded yet, and nothing else.

## Rebuild notes

- **All server state is in memory** (`MULTI-USER.md` §4.1). Restarting drops
  every in-flight job and its ownership records — the *files* here survive, but
  the job ids that reference them do not. Orphaned files accumulate; there is no
  reaper.
- **On a hosted / rented GPU, point this at persistent storage** if you care
  about surviving an instance restart:
  ```bash
  export SIZZLE_WORK_DIR=/workspace/sizzle-work
  ```
  `LOG_DIR` and `API_LOG_FILE` follow it automatically, or override separately
  with `SIZZLE_LOG_DIR` / `SIZZLE_API_LOG`.
- **Watch the disk.** Renders are the bulk of it and nothing is cleaned up
  automatically. On a rented box with a small root volume this fills up long
  before the weights do.
