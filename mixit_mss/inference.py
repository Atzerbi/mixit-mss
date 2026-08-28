"""Inference / separation utilities — importable from a notebook.

Two scenarios (see `separate`):
  - RAW pre-training model: emits the N (e.g. 12) unlabeled MixIT channels. Useful
    to *hear what the unsupervised model learned* (some channels catch vocals,
    others drums, possibly over-separated). Not a finished VDBO separator.
  - FINE-TUNED model: emits the 4 VDBO stems (vocals/drums/bass/other), using the
    channel_map saved by finetune_musdb.py.

Whole songs are separated with CHUNKED OVERLAP-ADD so they don't OOM the GPU: the
track is split into overlapping windows, each window is separated, and the outputs
are cross-faded back together.

Typical notebook use:
    from mixit_mss.inference import load_separator, separate, separate_musdb_test

    sep = load_separator("finetuned_musdb.pth", device="cuda")   # or a pretrain .pth
    stems = separate(sep, "song.wav")            # dict {name: waveform [C, L]}
    # or run over the whole MUSDB test split, writing wavs to disk:
    separate_musdb_test(sep, "/path/musdb18hq/test", "sep_out", limit=3)
"""

import os
import json
import math
import torch

from .separator_adapter import build_bslocoformer_adapter
from .channel_selection import STEMS


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class Separator:
    """Wraps an adapter model plus metadata (mode + optional channel map)."""
    def __init__(self, model, n_srcs, sr, channel_map=None, device="cpu"):
        self.model = model
        self.n_srcs = n_srcs
        self.sr = sr
        self.channel_map = channel_map      # dict stem->channel index, or None
        self.device = device
        # number of audio channels (2 = stereo); read from the adapter when present
        self.n_channels = getattr(model, "C", 2)

    @property
    def is_finetuned(self):
        return self.channel_map is not None


def load_separator(ckpt_path, device="cuda", n_srcs=12, n_channels=2,
                   n_layers=6, emb_dim=128, sr=44100, stft_size=2048, hop_length=512):
    """Load a checkpoint saved by either pretrain_mixit.py or finetune_musdb.py.

    Detects the format automatically:
      - pretrain: a raw state_dict            -> RAW mode (N channels)
      - finetune: {"state_dict", "channel_map"} -> FINE-TUNED mode (4 stems)

    The model hyperparameters MUST match those used at training time (n_srcs,
    n_layers, emb_dim, stft_size, hop_length). Adjust if you trained with others.
    """
    model = build_bslocoformer_adapter(
        n_srcs=n_srcs, n_channels=n_channels, stereo=(n_channels == 2),
        n_layers=n_layers, emb_dim=emb_dim,
        sample_rate=sr, stft_size=stft_size, hop_length=hop_length).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    channel_map = None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
        channel_map = ckpt.get("channel_map")
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return Separator(model, n_srcs=n_srcs, sr=sr, channel_map=channel_map, device=device)


# ---------------------------------------------------------------------------
# Chunked overlap-add core
# ---------------------------------------------------------------------------
@torch.no_grad()
def separate_waveform(sep, wav, chunk_seconds=6.0, overlap=0.5):
    """Separate a waveform [C, L] with chunked overlap-add.

    Returns raw model outputs [N, C, L] (all N channels). Labeling/selection into
    stems is done by the callers below, depending on the mode.
    """
    device = sep.device
    C, L = wav.shape
    chunk = int(round(chunk_seconds * sep.sr))
    hop = max(1, int(round(chunk * (1.0 - overlap))))

    # a Hann window for smooth cross-fade between chunks (avoids edge clicks)
    win = torch.hann_window(chunk, periodic=True, device=device)

    out = torch.zeros(sep.n_srcs, C, L, device=device)
    norm = torch.zeros(L, device=device)                 # sum of windows for normalization

    start = 0
    while start < L:
        end = min(start + chunk, L)
        seg = wav[:, start:end].to(device)               # [C, seg_len]
        seg_len = seg.shape[1]
        if seg_len < chunk:                              # pad last chunk
            seg = torch.nn.functional.pad(seg, (0, chunk - seg_len))

        est = sep.model(seg.unsqueeze(0))                # [1, N, C, chunk]
        est = est[0]                                     # [N, C, chunk]

        w = win[:seg_len]
        out[:, :, start:end] += est[:, :, :seg_len] * w
        norm[start:end] += w

        if end == L:
            break
        start += hop

    norm = norm.clamp_min(1e-8)
    out = out / norm                                     # overlap-add normalization
    return out.cpu()                                     # [N, C, L]


# ---------------------------------------------------------------------------
# High-level: separate one file into a named dict of stems
# ---------------------------------------------------------------------------
def _load_audio(path, target_sr, n_channels=2):
    import torchaudio
    wav, sr = torchaudio.load(path)                      # [C, L]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] == 1 and n_channels == 2:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > n_channels:
        wav = wav[:n_channels]
    return wav


@torch.no_grad()
def separate(sep, audio_path, chunk_seconds=6.0, overlap=0.5):
    """Separate one audio file. Returns dict {name: waveform [C, L]}.

    - FINE-TUNED model: names are the 4 VDBO stems, picked via channel_map.
    - RAW model: names are 'src00'..'srcNN', all N channels.
    """
    wav = _load_audio(audio_path, sep.sr)
    raw = separate_waveform(sep, wav, chunk_seconds, overlap)   # [N, C, L]

    if sep.is_finetuned:
        stems = {}
        for name in STEMS:
            ch = sep.channel_map[name]
            stems[name] = raw[ch]
        return stems
    else:
        return {f"src{c:02d}": raw[c] for c in range(sep.n_srcs)}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def save_stems(stems, out_dir, sr=44100):
    """Write a dict {name: [C, L]} to out_dir/<name>.wav."""
    import torchaudio
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, wav in stems.items():
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        p = os.path.join(out_dir, f"{name}.wav")
        torchaudio.save(p, wav, sr)
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# MUSDB test split
# ---------------------------------------------------------------------------
@torch.no_grad()
def separate_musdb_test(sep, musdb_test_root, out_root, limit=None,
                        chunk_seconds=6.0, overlap=0.5, mixture_name="mixture.wav"):
    """Separate every track in the MUSDB test split and write stems to disk.

    Expects the standard MUSDB18-HQ layout: musdb_test_root/<track>/mixture.wav
    (plus the ground-truth stems, which are ignored here). Writes results to
    out_root/<track>/<stem>.wav.

    Returns a list of (track_name, out_dir) for the tracks processed.
    """
    tracks = sorted(d for d in os.listdir(musdb_test_root)
                    if os.path.isdir(os.path.join(musdb_test_root, d)))
    if limit is not None:
        tracks = tracks[:limit]

    done = []
    for ti, track in enumerate(tracks):
        mix_path = os.path.join(musdb_test_root, track, mixture_name)
        if not os.path.exists(mix_path):
            print(f"[skip] {track}: no {mixture_name}", flush=True)
            continue
        stems = separate(sep, mix_path, chunk_seconds, overlap)
        out_dir = os.path.join(out_root, track)
        save_stems(stems, out_dir, sr=sep.sr)
        done.append((track, out_dir))
        print(f"[{ti+1}/{len(tracks)}] {track} -> {out_dir} "
              f"({'VDBO' if sep.is_finetuned else str(sep.n_srcs)+' raw channels'})",
              flush=True)
    return done