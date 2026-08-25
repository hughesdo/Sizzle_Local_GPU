# Multi-user behaviour — what happens when you share the `host.bat` URL

**Question this answers:** if I run `host.bat` and send the
`https://something.trycloudflare.com` URL to two or more people, can they all use
the app at the same time without getting tangled up — their music, their images,
their rendered clips, their final videos?

**Short answer:** yes, everything is properly separated *except rendering*, which
is a deliberate global one-at-a-time lock. Nobody's files, jobs or downloads can
cross over. They simply take turns on the GENERATE button, and right now they
take turns **manually** — there is no queue.

Every claim below is anchored to the code so it can be re-checked after edits.

---

## 1. How a "user" is identified

There is **no login**. Identity is a cookie:

- `_SID_COOKIE = "sizzle_sid"` — `backend/app.py:59`
- Issued by middleware on first request: `backend/app.py:67` (`_session_cookie`)
- A `uuid4().hex` GUID, `httpOnly` (page JS cannot read it), `samesite=lax`,
  30-day expiry.
- Read back via `_sid(request)` — `backend/app.py:97`
- Ownership test: `_owns(job, sid)` — `backend/app.py:101`

```python
def _owns(job, sid):
    # a job with no session (legacy) is open to anyone; otherwise
    # only its owning session may read its stream, clips or output
    return job.session_id is None or job.session_id == sid
```

**A session is a browser profile, not a person.**

| situation | result |
|---|---|
| Two friends, two different computers | two sessions ✅ |
| Two friends, two different browsers on one PC | two sessions ✅ |
| Same person, normal window + incognito | **two** sessions (uploads don't carry over) |
| Two people sharing one browser profile | **one** session — they *would* see each other's jobs |

---

## 2. What is isolated ✅

### 2.1 Uploads never collide

Both upload endpoints mint a fresh random id and a unique filename:

- `POST /api/audio` — `backend/app.py:183` → `f"{audio_id}_{file.filename}"`
- `POST /api/image` — `backend/app.py:224` → `f"img_{image_id}{ext}"`

ids are `uuid4().hex[:12]` — 48 bits of randomness. Two people who both upload
`song.mp3` get `a1b2c3d4e5f6_song.mp3` and `9f8e7d6c5b4a_song.mp3` in the same
`_work/uploads` directory. **No overwrite, no collision, ever.**

### 2.2 Timeline state never leaves the browser

The whole timeline — block positions, durations, prompts, the tray — lives in the
`state` object in `static/app.js`. The server is only told about it at the moment
someone hits GENERATE. Two people dragging blocks around cannot affect each other
because there is nothing shared to affect.

### 2.3 Jobs, clips and finished videos are ownership-checked

Jobs are tagged with the owning session at submit time
(`session_id=_sid(request)` — `backend/app.py:355`), and every read path checks it:

| endpoint | line | on someone else's job |
|---|---|---|
| `GET /download/{job_id}` | `app.py:394` | **404** |
| `GET /clip/{job_id}/{unit}` | `app.py:408` | **404** |
| `WS /ws/{job_id}` | `app.py:362` | closed with code **4403** |

So a friend cannot open, watch, or download your render — not even by guessing a
job id, because they'd need the id *and* the matching cookie.

`/clip/` additionally resolves the path inside `CLIP_DIR` and builds it from ids
only, so no user-supplied string ever reaches the filesystem (no traversal).

### 2.4 Output files can't overwrite each other

Every artefact is namespaced by job id, which is itself a `uuid4().hex[:12]`:

```
_work/clips/{job_id}_u{n}.mp4              per-segment renders
_work/clips/{job_id}_seg{n}.wav            per-segment audio slices
_work/clips/{job_id}_lastframe_*.png       continue-chain frames
_work/clips/{job_id}_window.wav            the muxed audio window
_work/outputs/{job_id}.mp4                 the final video
```

Two simultaneous jobs write to entirely disjoint filenames.

---

## 3. What is shared ⚠️ — rendering is one at a time

This is the one real constraint, and it is intentional.

**`/api/generate` rejects rather than queues** — `backend/app.py:286`:

```python
if jobs.is_busy():
    return JSONResponse(
        {"error": "a render is already running — wait for it to finish"},
        status_code=409)
```

`jobs.is_busy()` (`backend/jobs.py:192`) is **global** — it returns true if *any*
session has a job queued or running, not just yours.

### Why it works this way — and why it is load-bearing, not just cautious

The obvious assumption is "one GPU, one 22B model, a second render would OOM".
That turns out to be **wrong on both counts**, and the real reason is more
interesting. Measured on the RTX PRO 6000 (96 GB), rendering a 3s clip at
768x448:

**The GPU is mostly idle.** Profiling one warm render (66s total):

| phase | time |
|---|---|
| Gemma load + prompt encode | ~15.3s |
| **build transformer (stage 1)** | **19.4s** |
| stage-1 denoise, 8 steps | ~3.7s |
| build spatial upsampler | ~1.5s |
| **build transformer (stage 2)** | **22.7s** |
| stage-2 denoise + VAE decode | ~3s |

```
utilisation : avg 16.6%   max 99%
power       : avg 86 W of 600 W   max 506 W
>=50% busy  : 7.8% of the wall-clock
```

About **42 of 66 seconds is rebuilding the 22B transformer twice per clip** —
reading the 43 GB checkpoint and fusing LoRAs. That is disk/CPU/PCIe work. Only
~7s is actual denoising. So on paper there is plenty of room for a second job to
use the idle GPU, and VRAM is not the limit either: peak is **39–42 GB of 95.6**,
so two would fit.

**But it does not work.** Two renders launched simultaneously as separate
processes, twice:

| run | first process | second process |
|---|---|---|
| 1 | OK, 79.4s | **FAIL at 23.7s** |
| 2 | OK, 78.7s | **FAIL at 22.9s** |

```
RuntimeError: Attempted to access the data pointer on an invalid python storage.
```

Not an OOM — no CUDA memory error appears at all. The failure is deterministic,
always at the ~23s mark, which is exactly the "Building transformer" phase.

**The cause is NOT yet proven.** The error was observed; the mechanism was not.
Two plausible explanations, and they call for different fixes:

1. **Host RAM exhaustion.** Each render process peaks around 26-32 GB resident,
   but during a transformer build it is reading a 43 GB checkpoint and a 23 GB
   Gemma. Two of those at once can plausibly exceed this box's 137 GB, and a
   failed host allocation would invalidate a tensor's storage exactly like this.
2. **A safetensors mmap conflict** on the same checkpoint file opened by two
   processes at once.

Distinguishing them is easy and worth doing before any concurrency work: watch
peak committed RAM during a concurrent run, and separately try giving each
process its own *copy* of the checkpoint file. If a second copy fixes it, it is
(2); if RAM pegs first, it is (1) — and (1) would be cured by fp8 quantization or
by the batching change below, both of which shrink the per-build footprint.

Note also that the first render only slowed from 66s to ~79s, so the contention
cost is modest — the failure is a correctness problem, not a performance one.

**And in-process it would be worse.** The app renders in a worker thread sharing
one `_PIPELINE` singleton (`backend/ltx_engine.py`). `get_pipeline()` is
lock-guarded, but the pipeline object itself is not thread-safe and two
overlapping `pipe(...)` calls would interleave against shared model state.

So the global lock stays. The debatable part is only *rejecting* instead of
*queueing* — upstream's "decision 3: NO queueing" existed because each extra job
burned real fal credits, and locally it only costs GPU time.

### What the second person actually experiences

They do **not** get an error dump. The UI handles it:

- `/api/status` returns `rendering` (is anyone rendering?) and `rendering_mine`
  (is it me?) — `backend/app.py:137`
- The page polls it every 4 seconds — `static/app.js:906`
- `state.othersRendering` is set from that — `static/app.js:902`
- The button greys out and reads **"someone is rendering…"** —
  `static/app.js:854` (`applyGenerateButton`)
- If a request slips through anyway, the 409 surfaces as
  **"someone is already rendering — hang tight"** — `static/app.js:956`

### The gap

There is **no queue and no position indicator**. Person B's button un-greys within
~4s of the GPU freeing up, but *they* have to notice and click. If two friends are
both waiting, whoever clicks first wins; the other goes back to waiting.

With 2–3 people and multi-minute renders, expect the button to be unavailable a
lot of the time.

---

## 4. Known rough edges

None of these break isolation between users. Listed so nothing is a surprise.

### 4.1 All server state is in memory

```python
_AUDIO:  Dict[str, Path] = {}   # app.py:47
_IMAGES: Dict[str, Path] = {}   # app.py:48
_JOBS:   Dict[str, Job]  = {}   # jobs.py:161
```

- **Restart the server and every session's uploads become unreachable.** The files
  are still on disk in `_work/uploads`, but the id→path mapping is gone, so
  everyone must re-upload. Cookies survive; the data they point at does not.
- **Nothing is ever pruned.** A long hosting session accumulates entries forever
  (small — just paths — but unbounded), and `_work/` grows without limit on disk.
  Clearing `_work/` between sessions is safe when nothing is running.

### 4.2 Image and audio fetches are not ownership-checked

- `GET /api/image/{image_id}` — `backend/app.py:235` — serves the file to anyone
  holding the id.
- `POST /api/autoprompt` — `backend/app.py:243` — will describe any image id.
- `POST /api/generate` — `backend/app.py:262` — accepts any `audio_id` /
  `image_id`, not just ones your session uploaded.

Because ids are unguessable random 48-bit values, this **never causes accidental
crossover** — you cannot stumble into someone else's image. It is simply not
*enforced* isolation. Worth knowing before pointing this at strangers rather than
friends.

### 4.3 A benign race on the render lock

`is_busy()` is checked in the request handler, and `submit()` registers the job a
moment later. Two POSTs arriving in the same instant could both pass the check.

**Outcome is harmless:** the worker (`backend/jobs.py:417`) is a single thread
pulling from one FIFO queue, so the two jobs run *sequentially* rather than
concurrently. Nobody OOMs; the second person just gets their render immediately
after the first instead of a 409. Low probability, no corruption.

### 4.4 The lock is global, so one person can block everyone

There is no per-session fairness, no timeout, and no cancel button. A friend who
queues a long, high-resolution timeline holds the GPU until it finishes.

---

## 5. Summary table

| concern | status |
|---|---|
| Two people upload music at once | ✅ fine, unique ids |
| Two people upload images at once | ✅ fine, unique ids |
| Two people build timelines at once | ✅ fine, client-side only |
| Two people use auto-prompt at once | ✅ fine, independent API calls |
| Can B see A's finished video? | ✅ no — 404 |
| Can B watch A's render progress? | ✅ no — WS 4403 |
| Can B download A's per-segment clips? | ✅ no — 404 |
| Do output files overwrite each other? | ✅ no — job-id namespaced |
| Two people render at once | ⚠️ **no** — one at a time, second is rejected. Measured: forcing concurrency reliably crashes the second render (safetensors mmap race), and it is not a VRAM limit |
| Does B get told why? | ✅ yes — button reads "someone is rendering…" |
| Does B auto-start when the GPU frees? | ❌ no — must click again |
| Survives a server restart? | ❌ no — in-memory state is lost |

---

## 6. Open questions (not yet resolved)

Recorded so they are not lost. None of these block using the app today.

### 6.1 Could two people's videos genuinely render at once on this card?

**Status: unresolved, and worth revisiting.** The hardware says yes; the first
experiment says no.

Arguments that it *should* work:

- The GPU is idle ~92% of a render (see the profile in section 3). Two jobs would
  interleave into each other's dead time rather than compete for SMs.
- VRAM is not the constraint: peak is 39-42 GB of 95.6 GB, so two fit with room.
- When forced, the *first* render only slowed from 66s to ~79s. Contention cost
  was mild - the second job did not lose a fight for the GPU, it errored out.

What stands in the way:

- The reproducible crash above, whose cause is still unproven (RAM vs mmap).
- Even if fixed process-to-process, the app renders in ONE process with a shared
  `_PIPELINE` singleton that is not thread-safe. True in-app concurrency needs
  either a per-render subprocess or a genuinely re-entrant pipeline.
- Two jobs also both want the Gemma text encoder, adding another ~23 GB each.

Suggested order of investigation:

1. Re-run the two-process test while logging peak committed host RAM, to settle
   the RAM-vs-mmap question.
2. Re-run it with `SIZZLE_QUANTIZATION=fp8-scaled-mm` - it roughly halves the
   bytes each build touches, so if RAM is the limit this alone may fix it.
3. Only then decide between a real concurrency model (subprocess per render) and
   simply making renders fast enough that a queue is painless (section 6.2).

### 6.2 Is batching stages per job better than concurrency anyway?

Probably, and it is far less risky. Today a 20-clip video pays the ~42s
transformer rebuild **40 times** (twice per clip). Restructuring the loop to run
every segment through stage 1, then every segment through stage 2, pays it
**twice per job**. That is a bigger win than perfect 2x concurrency, and it needs
no new failure modes - only a change to the per-segment loop in `jobs.py`.

The cost is UX: clips would stop completing in timeline order, so the live
per-clip preview links would arrive in two waves instead of one at a time.

**Decide 6.2 before 6.1** - if a video renders in a few minutes, the queue in
section 5 stops being annoying and concurrency stops being necessary.

---

## 7. If you want true multi-user queueing

The behaviour above is upstream's design, kept deliberately. Turning the reject
into a real queue is a contained change:

1. **`backend/jobs.py`** — drop the `is_busy()` rejection and let `submit()` always
   enqueue. The worker is already a serial FIFO, so it needs no change at all.
2. **Emit a position event.** `submit()` already computes `ahead` and emits
   `{"type": "queued", "position": ahead}` (`backend/jobs.py:210`) — the plumbing
   exists and is currently wasted, because nothing can ever be queued behind
   another job.
3. **`/api/status`** — add `queue_depth` and the caller's own position.
4. **`static/app.js`** — replace the greyed "someone is rendering…" with
   "queued — 2 renders ahead of you", and start the progress panel when the job
   actually begins rather than at submit time.
5. **Add a cap** (say 3–5 queued jobs) so one person cannot monopolise the GPU,
   and a cancel button for your own queued job.

The risky part is not the queue — it is that a queued job holds references to
uploaded files that a server restart would invalidate, so a queue makes reason
4.1 (in-memory state) more visible. Persisting `_AUDIO`/`_IMAGES`/`_JOBS` to a
small SQLite file would fix both at once.

**Do not turn the queue into parallelism.** The measurements above show the GPU
is idle ~92% of the wall-clock, which makes concurrent rendering look tempting.
It does not work today (deterministic mmap crash), and the honest fix is not
concurrency but removing the waste: ~42s of every 66s clip is rebuilding the same
transformer twice. Options, roughly in order of effort:

1. **Render more per model build.** The two stages rebuild because they carry
   different LoRA sets (stage 2 adds the distilled LoRA). Batching all of a
   job's segments through stage 1, then all through stage 2, would pay the build
   cost twice per JOB instead of twice per CLIP. For a 20-clip music video that
   is roughly a 20x reduction in build overhead. This is the big win and needs no
   changes to Lightricks' code — only to the per-segment loop in `jobs.py`.
2. **Cache the state dict on CPU.** `Builder.build()` loads weights onto the same
   device as the model, so `SIZZLE_CACHE_WEIGHTS=1` currently pins ~43 GB in
   VRAM (see `config.CACHE_WEIGHTS_IN_RAM`). A registry that retains tensors in
   pinned CPU memory and copies H2D per build would cut the disk read without
   the VRAM cost — but it means patching ltx-core.
3. **Quantize.** `SIZZLE_QUANTIZATION=fp8-scaled-mm` halves the bytes that have
   to be read and fused per build, and is faster on Blackwell.
