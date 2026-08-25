# `static/` — the front-end

A single-page timeline editor. **No build step, no bundler, no npm, no
framework** — three hand-written files served directly by FastAPI
(`app.py` mounts this directory).

| File | Lines | What it is |
|---|---:|---|
| `index.html` | 157 | The whole page. Four `<section class="step">` blocks: music → images → timeline → render. |
| `app.js` | 1085 | All the behaviour: uploads, waveform, drag-and-drop timeline, prompt editing, WebSocket progress. Plain ES modules, no dependencies. |
| `style.css` | 349 | The look — dark, scanline, monospace. No preprocessor. |

**Rebuild note: this needs nothing installed.** Edit and reload. If you were
looking for a `package.json`, there isn't one and that is intentional.

## The core idea

**Images own time.** Each block's *width on the timeline is its render
duration*. Drag to place, drag the right edge to resize, click to edit its
prompt. Detected beats are soft snap guides, not a grid.

Two kinds of block:

- **image** — a first-frame image + prompt, rendered for its width.
- **continue** — no image of its own. Its first frame is the *last frame of the
  previous block*, so the video flows straight on. Chains of continues are
  allowed, which is how a whole video grows from one seed image.

The timeline may be **partial**: blocks need not cover the whole track. The
server renders only `[firstBlock.start, lastBlock.end]` and fills any gap
*inside* that window with black of the exact length, so the audio overlay never
desyncs.

All timeline state lives in the `state` object in `app.js` and **never leaves
the browser** — see `MULTI-USER.md` §2.2. A refresh loses your timeline; that is
current behaviour, not a bug to be surprised by.

## The contract with the API

`app.js` talks to `backend/app.py`. Change one side and you must change the
other — there is no schema or codegen keeping them honest.

| Call | Used for |
|---|---|
| `POST /api/audio` | upload music → waveform peaks, beats, downbeats, tempo, proposed segments |
| `POST /api/audio/manual` | re-segment with explicit user markers |
| `POST /api/image` | upload one image → id + thumb url |
| `POST /api/autoprompt` | image id → a suggested reactive prompt |
| `POST /api/generate` | submit the timeline → job id |
| `WS /ws/{job_id}` | live progress while rendering |
| `GET /download/{job_id}` | the finished muxed mp4 |
| `GET /api/status` | the three header badges + presets + limits |

**`/api/status` is what makes the UI self-configuring**, and it is the reason
the front-end has no GPU assumptions baked in. It supplies the format presets
(with megapixels), `minDim`/`maxDim`, `fps`, and the max clip length, all
derived server-side from `backend/config.py`. Change `SIZZLE_WIDTH`,
`SIZZLE_MAX_FRAMES` or the preset list on a new box and the dropdowns follow
automatically — no front-end edit needed.

The header badges read `render:`, `auto-prompt:` and `ffmpeg:`. On a fresh
rebuild you will see `render: local (weights missing)` until
`scripts/check_weights.py` is satisfied. That is the fastest visual confirmation
that a new box is wired up correctly.

## Constraints mirrored from the model

The UI enforces these client-side so it can never place a block the model would
reject — but `backend/config.py` is the authority and re-validates everything:

- **Resolution snaps to a 64-pixel grid** (`step="64"` on the custom W/H inputs,
  bounded 256–1920). Stage 1 renders at half size and the upscaler doubles it,
  so `ltx` asserts on anything off-grid.
- **Clip length is capped** at `MAX_CLIP_SECONDS` (19.04 s by default), because
  frame counts must be `8n+1`. The server snaps whatever it receives.

## Rebuild notes

- `index.html` links `style.css?v=5` — a **manual** cache-buster. Bump it when
  you change the CSS, or your browser will serve the old file and you will
  debug a stylesheet that isn't running.
- No GPU, model or platform assumptions live here. This directory is identical
  on every host.
- The only per-host thing worth knowing: if you serve over a Cloudflare tunnel
  (`host.bat` / `host.sh`), the WebSocket at `/ws/{job_id}` must survive the
  proxy. Quick tunnels handle this fine; a hand-rolled reverse proxy needs
  explicit WebSocket upgrade headers or progress will silently never arrive
  while renders still complete.
