"""FMA preprocessing and MoM loader — exact recipe from Saijo & Bando (Sec. 4.1, 4.3).

Two levels, faithful to the paper:

PRE-PROCESSING (offline, Sec. 4.1)
  - Segment each track into 10-second clips with 5-second overlap.
  - For each segment, compute signal power in 1-second intervals; DISCARD the
    segment if it contains more than 5 seconds of silence.
  - Sample-rate policy: drop any audio whose sample rate is below 44.1 kHz;
    downsample anything above 44.1 kHz to 44.1 kHz.
  This produces a manifest of valid 10 s segments (paths + offsets).

TRAINING (online, Sec. 4.3)
  - The training input is 6 seconds long (NOT 10): a 6 s window is randomly
    cropped from each valid 10 s segment at load time.
  - A MoM is x1 + x2 from two different segments (ideally different tracks).
  - Stereo (M=2). Negative thresholded SNR is the loss (implemented in losses.py).

The manifest step is separated so it runs once; MoMFMADataset then streams from it.
"""

import os
import json
import glob
import math
import random
import torch


# ---------------------------------------------------------------------------
# Recipe constants (from the paper)
# ---------------------------------------------------------------------------
TARGET_SR = 44100          # Sec. 4.1: target sample rate
CLIP_SECONDS = 10.0        # Sec. 4.1: segment length for pre-processing
CLIP_OVERLAP = 5.0         # Sec. 4.1: overlap between segments
SILENCE_INTERVAL = 1.0     # Sec. 4.1: power computed in 1 s intervals
MAX_SILENCE_SECONDS = 5.0  # Sec. 4.1: discard if > 5 s of silence
TRAIN_INPUT_SECONDS = 6.0  # Sec. 4.3: training input length


def _is_silent_interval(power, threshold):
    return power < threshold


def scan_track_for_segments(wav, sr, silence_threshold,
                            clip_s=CLIP_SECONDS, overlap_s=CLIP_OVERLAP,
                            interval_s=SILENCE_INTERVAL,
                            max_silence_s=MAX_SILENCE_SECONDS):
    """Given a loaded waveform [C, N] at sample rate `sr`, return the list of
    valid segment start-sample offsets according to the paper's silence rule.

    A segment is valid iff it contains at most `max_silence_s` seconds of silence,
    measured over non-overlapping `interval_s` windows.
    """
    C, N = wav.shape
    clip = int(round(clip_s * sr))
    hop = int(round((clip_s - overlap_s) * sr))
    interval = int(round(interval_s * sr))
    max_silent_intervals = int(math.floor(max_silence_s / interval_s))

    # per-interval power over the whole track (mono-averaged), computed once
    mono = wav.mean(dim=0)                                  # [N]
    n_intervals = N // interval
    if n_intervals == 0:
        return []
    trimmed = mono[: n_intervals * interval].reshape(n_intervals, interval)
    interval_power = (trimmed ** 2).mean(dim=1)             # [n_intervals]

    valid = []
    start = 0
    while start + clip <= N:
        # which power-intervals fall inside this clip
        i0 = start // interval
        i1 = (start + clip) // interval
        seg_powers = interval_power[i0:i1]
        n_silent = int(_is_silent_interval(seg_powers, silence_threshold).sum().item())
        if n_silent <= max_silent_intervals:
            valid.append(start)
        start += hop
    return valid


def build_manifest(clip_dir, out_path, silence_threshold=1e-4,
                   target_sr=TARGET_SR, exts=("*.wav", "*.flac", "*.mp3", "*.ogg"),
                   verbose=True):
    """Offline pre-processing pass. Scans `clip_dir`, applies the sample-rate policy
    and the silence rule, and writes a JSON manifest of valid 10 s segments.

    Manifest entry: {"path": str, "offset": int(samples @ target_sr), "sr": target_sr}

    Requires torchaudio. Tracks below target_sr are dropped; above are resampled.
    """
    import torchaudio

    files = sorted(sum((glob.glob(os.path.join(clip_dir, "**", e), recursive=True)
                        for e in exts), []))
    entries = []
    dropped_sr = 0
    for fi, path in enumerate(files):
        try:
            info = torchaudio.info(path)
        except Exception:
            continue
        orig_sr = info.sample_rate
        if orig_sr < target_sr:
            dropped_sr += 1
            continue                                  # Sec. 4.1: drop < 44.1 kHz
        wav, sr = torchaudio.load(path)               # [C, N]
        if sr > target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)  # downsample
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)                     # force stereo view for scan
        offsets = scan_track_for_segments(wav, target_sr, silence_threshold)
        for off in offsets:
            entries.append({"path": path, "offset": int(off), "sr": target_sr,
                            "orig_sr": int(orig_sr)})
        if verbose and (fi + 1) % 200 == 0:
            print(f"scanned {fi+1}/{len(files)} files, {len(entries)} segments", flush=True)

    with open(out_path, "w") as f:
        json.dump({"target_sr": target_sr,
                   "clip_seconds": CLIP_SECONDS,
                   "overlap_seconds": CLIP_OVERLAP,
                   "silence_threshold": silence_threshold,
                   "segments": entries}, f)
    if verbose:
        print(f"manifest: {len(entries)} valid segments "
              f"({dropped_sr} files dropped for sr < {target_sr}) -> {out_path}",
              flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Online MoM dataset streaming from a manifest
# ---------------------------------------------------------------------------
class MoMFMADataset(torch.utils.data.Dataset):
    """Mixture-of-Mixtures over FMA valid segments (paper recipe).

    At each __getitem__:
      - load a 6 s random crop from segment `idx` (x1),
      - load a 6 s random crop from another segment, preferably a different track (x2),
      - return mom = x1 + x2  [2, L6],  mixtures = stack([x1, x2])  [2, 2, L6].
    """

    def __init__(self, manifest_path=None, manifest=None,
                 train_input_seconds=TRAIN_INPUT_SECONDS, target_sr=TARGET_SR,
                 avoid_same_track=True, rms_normalize=True, _synthetic=False):
        self.sr = target_sr
        self.L = int(round(train_input_seconds * target_sr))
        self.clip_len = int(round(CLIP_SECONDS * target_sr))
        self.avoid_same_track = avoid_same_track
        self.rms_normalize = rms_normalize
        self._synthetic = _synthetic

        if _synthetic:
            # fabricate a tiny manifest so the pipeline runs without data
            self.segments = [{"path": f"synthetic_{i}", "offset": 0, "sr": target_sr}
                             for i in range(64)]
            return

        if manifest is None:
            with open(manifest_path) as f:
                manifest = json.load(f)
        self.segments = manifest["segments"]
        if len(self.segments) == 0:
            raise ValueError("Empty manifest: no valid FMA segments")

    def __len__(self):
        return len(self.segments)

    def _load_crop(self, seg):
        """Load a 6 s stereo crop [2, L] from a 10 s segment entry."""
        if self._synthetic:
            return torch.randn(2, self.L)
        import torchaudio
        # load exactly the 10 s segment window, then random 6 s sub-crop
        num_frames = self.clip_len
        wav, sr = torchaudio.load(seg["path"], frame_offset=seg["offset"],
                                  num_frames=num_frames)      # [C, <=clip_len]
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        # random 6 s sub-crop within the (up to) 10 s window
        if wav.shape[1] < self.L:
            wav = torch.nn.functional.pad(wav, (0, self.L - wav.shape[1]))
        start = random.randint(0, wav.shape[1] - self.L)
        crop = wav[:, start:start + self.L]                   # [2, L]
        if self.rms_normalize:
            crop = crop / (crop.std() + 1e-8)
        return crop

    def _pick_other(self, idx):
        if len(self.segments) == 1:
            return idx
        path_i = self.segments[idx]["path"]
        for _ in range(8):
            j = random.randrange(len(self.segments))
            if not self.avoid_same_track:
                return j
            if self.segments[j]["path"] != path_i:
                return j
        return j  # give up after a few tries

    def __getitem__(self, idx):
        x1 = self._load_crop(self.segments[idx])
        x2 = self._load_crop(self.segments[self._pick_other(idx)])
        mom = x1 + x2                                          # [2, L]
        mixtures = torch.stack([x1, x2], dim=0)               # [2, 2, L]
        return mom, mixtures