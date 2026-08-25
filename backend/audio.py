"""
Audio analysis for the timeline.

Two jobs:
  1. Produce a waveform peak array so the front-end can draw the track.
  2. Propose segment boundaries (<= MAX_CLIP_SECONDS), optionally snapped to
     detected beats, so each segment maps to one image + one LTX render.

Segments are just (start, end) float pairs in seconds. The front-end lets the
user drop an image onto each segment; the render loop turns each into a clip.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import librosa
import numpy as np

from . import config


@dataclass
class Segment:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def num_frames(self) -> int:
        # frames for this segment, snapped to a valid LTX count (F=8n+1) and
        # clamped to the model's [MIN_FRAMES, MAX_FRAMES] window.
        return config.frames_for(self.duration)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        d["num_frames"] = self.num_frames
        return d


@dataclass
class AudioAnalysis:
    duration: float
    sample_rate: int
    tempo: float
    peaks: List[float]           # downsampled abs-amplitude envelope, 0..1
    segments: List[Segment]
    beats: List[float]           # detected beat times (s) — soft snap guides
    downbeats: List[float]       # stronger beats (bar starts, best-effort)

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "tempo": round(float(self.tempo), 2),
            "peaks": [round(p, 4) for p in self.peaks],
            "segments": [s.to_dict() for s in self.segments],
            "beats": [round(float(b), 3) for b in self.beats],
            "downbeats": [round(float(b), 3) for b in self.downbeats],
        }


def _waveform_peaks(y: np.ndarray, buckets: int = 1200) -> List[float]:
    """Downsample the signal to `buckets` peak values in 0..1 for drawing."""
    if len(y) == 0:
        return []
    n = min(buckets, len(y))
    edges = np.linspace(0, len(y), n + 1, dtype=int)
    peaks = np.zeros(n, dtype=np.float32)
    for i in range(n):
        chunk = y[edges[i]:edges[i + 1]]
        if len(chunk):
            peaks[i] = float(np.max(np.abs(chunk)))
    m = float(peaks.max()) or 1.0
    return (peaks / m).tolist()


def _segments_fixed(duration: float) -> List[Segment]:
    """Uniform chunks capped at MAX_CLIP_SECONDS."""
    segs: List[Segment] = []
    t = 0.0
    idx = 0
    while t < duration - 1e-3:
        end = min(t + config.MAX_CLIP_SECONDS, duration)
        segs.append(Segment(idx, t, end))
        idx += 1
        t = end
    return segs


def _segments_from_beats(y: np.ndarray, sr: int, duration: float) -> List[Segment]:
    """
    Group detected beats into segments that stay under MAX_CLIP_SECONDS and
    above MIN_CLIP_SECONDS. Boundaries land on beats so motion resets on a beat.
    """
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Always include 0 and the track end as candidate cut points.
    cuts = [0.0] + [t for t in beat_times if 0.0 < t < duration] + [duration]
    cuts = sorted(set(round(c, 3) for c in cuts))

    # Walk beats, closing a segment at the last beat that keeps it under the cap.
    # Any span longer than the cap (sparse/no beats) is hard-split into <=cap
    # pieces so a segment can NEVER exceed MAX_CLIP_SECONDS.
    segs: List[Segment] = []
    seg_start = cuts[0]
    prev = cuts[0]
    for c in cuts[1:]:
        if c - seg_start > config.MAX_CLIP_SECONDS:
            # closing point is the previous beat if it made a usable segment,
            # otherwise we're forced to hard-split the gap at the cap.
            if prev > seg_start + 1e-3:
                seg_end = prev
            else:
                seg_end = seg_start + config.MAX_CLIP_SECONDS
            # hard-split anything still over cap (covers giant beatless gaps)
            t = seg_start
            while seg_end - t > config.MAX_CLIP_SECONDS + 1e-6:
                nt = t + config.MAX_CLIP_SECONDS
                segs.append(Segment(len(segs), t, nt))
                t = nt
            segs.append(Segment(len(segs), t, seg_end))
            seg_start = seg_end
        prev = c

    # flush the tail, hard-splitting if the remainder exceeds the cap
    t = seg_start
    while duration - t > config.MAX_CLIP_SECONDS + 1e-6:
        nt = t + config.MAX_CLIP_SECONDS
        segs.append(Segment(len(segs), t, nt))
        t = nt
    if duration - t > 1e-3:
        segs.append(Segment(len(segs), t, duration))
    # reindex
    segs = [Segment(i, s.start, s.end) for i, s in enumerate(segs)]

    # merge sub-minimum stubs into the previous segment
    merged: List[Segment] = []
    for s in segs:
        if merged and s.duration < config.MIN_CLIP_SECONDS:
            prev = merged[-1]
            # only merge if it keeps us under the cap
            if (s.end - prev.start) <= config.MAX_CLIP_SECONDS:
                merged[-1] = Segment(prev.index, prev.start, s.end)
                continue
        merged.append(Segment(len(merged), s.start, s.end))
    return merged or _segments_fixed(duration)


def _detect_beats(y: np.ndarray, sr: int, duration: float):
    """Return (tempo, beat_times, downbeat_times).

    Beats are the raw detected beat grid. Downbeats are a best-effort subset —
    every 4th beat (typical 4/4 bar) as a stronger visual guide. These drive the
    timeline tick marks; they are guides for snapping, never hard cut points.
    """
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        beat_times = [t for t in librosa.frames_to_time(beat_frames, sr=sr).tolist()
                      if 0.0 <= t <= duration]
    except Exception:
        tempo, beat_times = 0.0, []

    # Best-effort downbeats: phase-align 4/4 bars to the strongest onset beat.
    downbeats: List[float] = []
    if beat_times:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            times = librosa.times_like(onset_env, sr=sr)
            # strength sampled at each beat, pick the phase (0..3) with max energy
            def strength_at(t):
                i = int(np.searchsorted(times, t))
                i = min(max(i, 0), len(onset_env) - 1)
                return float(onset_env[i])
            best_phase, best_sum = 0, -1.0
            for phase in range(4):
                s = sum(strength_at(beat_times[i]) for i in range(phase, len(beat_times), 4))
                if s > best_sum:
                    best_sum, best_phase = s, phase
            downbeats = [beat_times[i] for i in range(best_phase, len(beat_times), 4)]
        except Exception:
            downbeats = beat_times[::4]
    return tempo, beat_times, downbeats


def analyze(path: Path, mode: str = "beat") -> AudioAnalysis:
    """
    mode: "beat" (snap to detected beats) or "fixed" (uniform 6s chunks).
    Manual-marker mode is handled front-end side; it POSTs explicit boundaries.
    """
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration = float(len(y) / sr) if sr else 0.0

    tempo, beat_times, downbeats = _detect_beats(y, sr, duration)

    if mode == "fixed":
        segments = _segments_fixed(duration)
    else:
        segments = _segments_from_beats(y, sr, duration)

    return AudioAnalysis(
        duration=duration,
        sample_rate=sr,
        tempo=tempo,
        peaks=_waveform_peaks(y),
        segments=segments,
        beats=beat_times,
        downbeats=downbeats,
    )


def slice_audio(path: Path, start: float, end: float, out_path: Path) -> Path:
    """Write the [start, end] slice of the source audio to out_path (wav).

    The slice is ALWAYS exactly (end - start) seconds long, silence-padded if the
    window runs past the end of the track. That matters because the local LTX
    pipeline derives the audio latent shape from num_frames/fps and then trims
    the encoded audio to it: hand it a slice that is even slightly short - which
    is what the final block of any timeline that reaches the end of the song
    produces - and the audio latent comes up short of the shape the denoiser
    expects. Padding here keeps every segment's audio exactly as long as the
    video it drives, which is the same invariant the mux relies on.
    """
    import soundfile as sf

    y, sr = librosa.load(str(path), sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    a = max(0, int(start * sr))
    b = int(end * sr)
    want = max(0, b - a)
    clip = y[:, a:b]
    if clip.shape[1] < want:
        pad = np.zeros((clip.shape[0], want - clip.shape[1]), dtype=clip.dtype)
        clip = np.concatenate([clip, pad], axis=1)
    sf.write(str(out_path), clip.T, sr)
    return out_path
