"""Datasets for the two stages.

MoMDataset   : mixture-of-mixtures from UNLABELED audio (FMA) for MixIT pre-training.
               No ground truth: input = x1+x2, target = [x1, x2].
MUSDBDataset : VDBO stems from MUSDB18 for supervised fine-tuning (PIT).

Real audio loading (torchaudio) is isolated in _load_segment/_load_stems, flagged
with TODO. The defaults generate noise so tests run without data.
"""

import os
import glob
import json
import random
import torch


# ---------------------------------------------------------------------------
# Pre-training: Mixture-of-Mixtures from unlabeled data
# ---------------------------------------------------------------------------
class MoMDataset(torch.utils.data.Dataset):
    def __init__(self, clip_dir=None, clip_paths=None, segment_len=441000,
                 n_channels=2, sr=44100, avoid_same_clip=True, _synthetic=False):
        """
        clip_dir     : folder of audio files (glob *.wav/*.flac/*.mp3), or
        clip_paths   : explicit list of paths.
        segment_len  : L samples (441000 = 10 s @ 44.1 kHz, as in the paper).
        _synthetic   : if True, generate noise instead of reading files (for tests).
        """
        self.L = segment_len
        self.C = n_channels
        self.sr = sr
        self.avoid_same_clip = avoid_same_clip
        self._synthetic = _synthetic

        if _synthetic:
            self.clips = [f"synthetic_{i}" for i in range(64)]
        elif clip_paths is not None:
            self.clips = list(clip_paths)
        elif clip_dir is not None:
            exts = ("*.wav", "*.flac", "*.mp3", "*.ogg")
            self.clips = sorted(sum((glob.glob(os.path.join(clip_dir, "**", e),
                                               recursive=True) for e in exts), []))
        else:
            raise ValueError("Provide clip_dir, clip_paths, or _synthetic=True")
        if len(self.clips) == 0:
            raise ValueError("No clips found")

    def __len__(self):
        return len(self.clips)

    def _load_segment(self, path):
        """Loads a segment [C, L].
        TODO(real): torchaudio.load(path) -> resample to self.sr -> random crop
        to self.L -> drop if >5s of silence (paper recipe) -> RMS-normalize.
        """
        if self._synthetic:
            return torch.randn(self.C, self.L)
        # --- real implementation (skeleton) ---
        import torchaudio
        wav, sr = torchaudio.load(path)                       # [ch, n]
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] == 1 and self.C == 2:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > self.C:
            wav = wav[:self.C]
        # random crop of length L (pad if too short)
        if wav.shape[1] < self.L:
            wav = torch.nn.functional.pad(wav, (0, self.L - wav.shape[1]))
        start = random.randint(0, wav.shape[1] - self.L)
        seg = wav[:, start:start + self.L]
        seg = seg / (seg.std() + 1e-8)                        # RMS-normalize
        return seg

    def __getitem__(self, idx):
        x1 = self._load_segment(self.clips[idx])
        j = idx
        if len(self.clips) > 1:
            while j == idx and self.avoid_same_clip:
                j = random.randrange(len(self.clips))
        x2 = self._load_segment(self.clips[j])
        mom = x1 + x2                                          # [C, L]
        mixtures = torch.stack([x1, x2], dim=0)               # [2, C, L]
        return mom, mixtures


# ---------------------------------------------------------------------------
# Fine-tuning: supervised MUSDB18 (VDBO stems)
# ---------------------------------------------------------------------------
STEMS = ("vocals", "drums", "bass", "other")


class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, segment_len=441000, n_channels=2, sr=44100,
                 dynamic_mixing=True, sad_manifest=None, drop_prob=0.05,
                 rms_normalize=True, _synthetic=False):
        """
        root           : MUSDB18 root (per-track subfolders with the stems).
        dynamic_mixing : recombine stems from different tracks (paper augmentation).
        sad_manifest   : path to a SAD manifest (mixit_mss.sad.build_musdb_sad_manifest)
                         or the loaded dict. If given, stem segments are sampled ONLY
                         from salient (non-silent) regions per the paper.
        drop_prob      : per-stem drop probability (paper: 0.05).
        rms_normalize  : RMS-normalize each stem before gain (paper recipe).
        _synthetic     : noise instead of files, for tests.
        Returns: mix [C, L], targets [4, C, L] in STEMS order.
        """
        self.L = segment_len
        self.C = n_channels
        self.sr = sr
        self.dynamic_mixing = dynamic_mixing
        self.drop_prob = drop_prob
        self.rms_normalize = rms_normalize
        self._synthetic = _synthetic

        # optional SAD manifest: {track: {stem: [offsets]}}
        self.sad = None
        if sad_manifest is not None and not _synthetic:
            if isinstance(sad_manifest, str):
                with open(sad_manifest) as f:
                    sad_manifest = json.load(f)
            # index by stem -> list of (track_path, offset)
            self.sad = {s: [] for s in STEMS}
            for entry in sad_manifest["tracks"]:
                for s, offs in entry["stem_offsets"].items():
                    for off in offs:
                        self.sad[s].append((entry["track"], off))

        if _synthetic:
            self.tracks = [f"synthetic_track_{i}" for i in range(32)]
        else:
            self.tracks = sorted(glob.glob(os.path.join(root, "*"))) if root else []
            if len(self.tracks) == 0:
                raise ValueError("No MUSDB tracks found")

    def __len__(self):
        return len(self.tracks)

    def _load_stem_segment(self, stem):
        """Load one [C, L] segment for `stem`.
        If a SAD manifest is present, sample a salient (track, offset) for this stem;
        otherwise fall back to a random track + random crop.
        """
        if self._synthetic:
            return torch.randn(self.C, self.L)
        import torchaudio
        if self.sad is not None and len(self.sad[stem]) > 0:
            track, offset = random.choice(self.sad[stem])
            path = os.path.join(track, f"{stem}.wav")
            wav, sr = torchaudio.load(path, frame_offset=offset, num_frames=self.L)
        else:
            track = random.choice(self.tracks)
            path = os.path.join(track, f"{stem}.wav")
            wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[1] < self.L:
            wav = torch.nn.functional.pad(wav, (0, self.L - wav.shape[1]))
        if wav.shape[1] > self.L:
            start = random.randint(0, wav.shape[1] - self.L)
            wav = wav[:, start:start + self.L]
        return wav[:self.C]

    def _augment(self, stem):
        """Paper's dynamic mixing: RMS-normalize, random gain [-10,10] dB,
        drop stem with probability drop_prob (0.05)."""
        if self.rms_normalize:
            stem = stem / (stem.std() + 1e-8)
        gain_db = random.uniform(-10, 10)
        stem = stem * (10 ** (gain_db / 20))
        if random.random() < self.drop_prob:
            stem = torch.zeros_like(stem)
        return stem

    def __getitem__(self, idx):
        stems = {}
        for s in STEMS:
            if self.dynamic_mixing:
                stems[s] = self._load_stem_segment(s)   # per-stem salient/random sample
            else:
                stems[s] = self._load_stem_segment(s)
        if self.dynamic_mixing:
            stems = {s: self._augment(v) for s, v in stems.items()}

        targets = torch.stack([stems[s] for s in STEMS], dim=0)   # [4, C, L]
        mix = targets.sum(dim=0)                                   # [C, L]
        return mix, targets