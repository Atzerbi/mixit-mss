"""Unsupervised energy-based Source Activity Detection (SAD).

Exact recipe from BSRNN (Luo & Yu, Sec. IV-A1), which the Saijo & Bando paper
applies to the MUSDB training stems to remove silent regions before dynamic mixing.

Algorithm, given an unsegmented track and a segment length L (seconds):
  1. split the track into overlapping segments of length L with 50% overlap;
  2. split each segment into 10 chunks of length L/10, compute each chunk's energy;
  3. for silent chunks, set energy to eps = 1e-5;
  4. compute a per-track energy threshold =
        max( 15%-quantile of all chunk energies over the track , gamma=1e-3 );
  5. a segment is SALIENT (kept) iff more than 50% of its 10 chunks have energy
     above the threshold.
Saijo & Bando use L = 6 seconds.

The threshold is computed per full track (step 4 uses all chunks of the track),
so this is a track-level scan producing a list of valid segment offsets, exactly
like the FMA manifest but with an energy-quantile criterion instead of a silence
duration criterion.
"""

import os
import glob
import json
import math
import torch


# Recipe constants (BSRNN Sec. IV-A1)
EPS_SILENT = 1e-5      # energy floor for silent chunks
GAMMA = 1e-3           # threshold floor
QUANTILE = 0.15        # 15% quantile of chunk energies
CHUNKS_PER_SEGMENT = 10
SALIENT_FRACTION = 0.5  # > 50% of chunks above threshold
OVERLAP_RATIO = 0.5    # 50% segment overlap
SEGMENT_SECONDS = 6.0  # L in the Saijo paper


def _chunk_energies(track_mono, chunk_len):
    """Energy per non-overlapping chunk over a whole mono track [N] -> [n_chunks]."""
    n = track_mono.shape[0] // chunk_len
    if n == 0:
        return torch.empty(0)
    trimmed = track_mono[: n * chunk_len].reshape(n, chunk_len)
    e = (trimmed ** 2).mean(dim=1)
    e = torch.where(e <= 0, torch.full_like(e, EPS_SILENT), e)
    return e


def sad_scan_track(wav, sr, segment_seconds=SEGMENT_SECONDS):
    """Return the list of salient segment start-sample offsets for a track.

    wav : [C, N] waveform. sr : sample rate.
    Follows the BSRNN energy-thresholding recipe exactly.
    """
    C, N = wav.shape
    seg_len = int(round(segment_seconds * sr))
    if N < seg_len:
        return []
    chunk_len = seg_len // CHUNKS_PER_SEGMENT
    hop = int(round(seg_len * (1.0 - OVERLAP_RATIO)))
    mono = wav.mean(dim=0)                                  # [N]

    # step 4: per-track threshold from the 15% quantile of ALL chunk energies
    all_e = _chunk_energies(mono, chunk_len)
    if all_e.numel() == 0:
        return []
    q = torch.quantile(all_e, QUANTILE)
    threshold = max(float(q), GAMMA)

    valid = []
    start = 0
    while start + seg_len <= N:
        seg = mono[start:start + seg_len]
        # step 2-3: energies of the 10 chunks in this segment
        e = _chunk_energies(seg, chunk_len)
        if e.numel() == CHUNKS_PER_SEGMENT:
            frac_above = float((e > threshold).float().mean().item())
            if frac_above > SALIENT_FRACTION:
                valid.append(start)
        start += hop
    return valid


def build_musdb_sad_manifest(root, out_path, segment_seconds=SEGMENT_SECONDS,
                             target_sr=44100, stems=("vocals", "drums", "bass", "other"),
                             verbose=True):
    """Offline SAD pass over MUSDB tracks. For each track and each stem, records the
    salient segment offsets (the paper applies SAD to the training stems).

    Manifest entry per track:
      {"track": path, "stem_offsets": {stem: [offset, ...]}, "sr": target_sr}
    """
    import torchaudio
    tracks = sorted(glob.glob(os.path.join(root, "*")))
    entries = []
    for ti, track in enumerate(tracks):
        stem_offsets = {}
        for s in stems:
            path = os.path.join(track, f"{s}.wav")
            if not os.path.exists(path):
                continue
            wav, sr = torchaudio.load(path)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            stem_offsets[s] = sad_scan_track(wav, target_sr, segment_seconds)
        entries.append({"track": track, "stem_offsets": stem_offsets, "sr": target_sr})
        if verbose and (ti + 1) % 20 == 0:
            print(f"SAD-scanned {ti+1}/{len(tracks)} tracks", flush=True)

    with open(out_path, "w") as f:
        json.dump({"segment_seconds": segment_seconds, "target_sr": target_sr,
                   "tracks": entries}, f)
    if verbose:
        total = sum(len(v) for e in entries for v in e["stem_offsets"].values())
        print(f"SAD manifest: {len(entries)} tracks, {total} salient stem-segments "
              f"-> {out_path}", flush=True)
    return out_path
