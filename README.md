# mixit-mss

**MixIT unsupervised pre-training** layer for music source separation (MSS), built on top of MERL's official **BS-Locoformer** (now vendored into this repo). Reproduces the scheme of *"Is MixIT Really Unsuitable for Correlated Sources? Exploring MixIT for Unsupervised Pre-training in Music Source Separation"* (Saijo & Bando): MixIT pre-training on the Free Music Archive (unlabeled) → channel selection → supervised fine-tuning on MUSDB18.

## What changed after integrating the real model

The BS-Locoformer standalone code from [`merlresearch/tf-locoformer`](https://github.com/merlresearch/tf-locoformer) (Apache-2.0) is now vendored under `mixit_mss/bslocoformer/` and wired into the pipeline. Reading the real source corrected several assumptions:

- **The model works entirely in the TF domain.** It takes a complex spectrogram `[B, C, T, F]` and returns complex spectrograms `[B, N, C, T, F]`; it does **not** perform STFT internally. So the (i)STFT is handled outside, in `mixit_mss/stft.py`.
- **Native stereo already exists** via the `stereo=True` flag — no decoder surgery needed, contrary to the earlier plan. With `stereo=True` the model ingests 2 channels and emits N sources with 2 channels each.
- **Band-split is Q=62 bands** out of the box (the model prints `Band-split module has 62 bands`), matching the paper's configuration.
- The output is a masked complex spectrogram; `SeparatorAdapter` runs STFT → BS-Locoformer → iSTFT and returns waveforms, so the MixIT loss (which compares waveforms) is unchanged.

## Design principle

This repo provides the **MixIT layer** that does not exist publicly on top of BS-Locoformer: the efficient MixIT loss, the mixture-of-mixtures dataset, channel selection, and the two-stage training loop. The separator is vendored unmodified (Apache-2.0) and attached through `SeparatorAdapter`, whose clean contract is `mom [B,C,L] -> sources [B,N,C,L]` (waveforms).

## Why a mask-based TF-domain model and not Demucs

MixIT needs **over-provisioned N outputs** (12 in the paper) relative to the 4 real stems: this is the mechanism by which the mixture-of-mixtures is separated. BS-Locoformer treats N (`num_spk`) as a hyperparameter; a fixed-stem model like Demucs does not. The band-split also matters for music, where sub-bands have very different statistics. MixIT remains pre-training only: the model is then fine-tuned with supervision, and that is the final separator.

## Structure

```
mixit_mss/
  bslocoformer/          # VENDORED real model (MERL, Apache-2.0), unmodified
    bslocoformer_separator.py
    tflocoformer_separator.py
  stft.py               # external STFT/iSTFT (model is TF-domain only)
  separator_adapter.py  # STFT -> BS-Locoformer -> iSTFT; real + stub builders
  losses.py             # efficient MixIT (Eq.3) + exhaustive (validation) + sparsity
  fma.py                # EXACT FMA recipe: segmentation + silence rule + MoM loader
  sad.py                # EXACT BSRNN source-activity detection (energy thresholding)
  datasets.py           # generic MoMDataset + MUSDBDataset (VDBO, SAD-aware)
  channel_selection.py  # maps the N MixIT channels onto the 4 VDBO stems (Sec. 4.3)
  pit.py                # PIT / direct loss for fine-tuning
scripts/
  build_fma_manifest.py # Stage 0a (offline: scan FMA, apply recipe, write manifest)
  build_musdb_sad.py    # Stage 0b (offline: SAD over MUSDB stems, write manifest)
  pretrain_mixit.py     # Stage 1 (real model by default; --stub to bypass)
  finetune_musdb.py     # Stage 2 (channel selection + fine-tuning)
configs/                # example YAML with the paper's recipe values
tests/
  test_pipeline.py      # shapes, backward, efficient-vs-exhaustive, real model
```

## Installation

```bash
pip install -r requirements.txt   # includes rotary-embedding-torch for the real model
```

## Quick check

Full suite, including an end-to-end pass through the **real BS-Locoformer** (small config):

```bash
PYTHONPATH=. python tests/test_pipeline.py
```

Two-stage run with the real model (small config, synthetic data):

```bash
# Stage 1: MixIT pre-training
PYTHONPATH=. python scripts/pretrain_mixit.py --synthetic --steps 4 \
    --segment_len 16384 --n_srcs 12 --n_layers 1 --emb_dim 32 --batch_size 1

# Stage 2: channel selection + supervised fine-tuning
PYTHONPATH=. python scripts/finetune_musdb.py --synthetic --pretrained pretrained_mixit.pth \
    --steps 4 --segment_len 16384 --n_srcs 12 --n_layers 1 --emb_dim 32 --batch_size 1
```

Add `--stub` to swap in a lightweight TF-domain stub instead of the real model (faster shape/flow checks). The `test_efficient_matches_exhaustive` test compares the least-squares solver against the exhaustive O(M^N) search: gap is 0 on the constructed case.

## Real-model configuration

`build_bslocoformer_adapter()` instantiates the vendored model. Key parameters and the paper-scale defaults:

- `n_srcs` → `num_spk` (12 for MixIT pre-training, over-provisioned vs 4 stems).
- `stereo=True` → native stereo I/O.
- `stft_size=2048`, `hop_length=512`, `sample_rate=44100` → the STFT config; `stft_size` must match the external STFT `n_fft`.
- `n_layers=6`, `emb_dim=128` → paper-scale capacity (reduce for prototyping).
- Band-split is fixed to the Q=62 configuration inside the model.

## FMA pre-training recipe (implemented, Sec. 4.1 & 4.3)

`mixit_mss/fma.py` implements the paper's exact recipe, in two levels:

**Offline pre-processing** (`build_fma_manifest.py` → `build_manifest`):
- segment each track into 10 s clips with 5 s overlap;
- compute signal power in 1 s intervals, discard a segment if it contains more than 5 s of silence;
- sample-rate policy: drop audio below 44.1 kHz, downsample anything above to 44.1 kHz;
- writes a JSON manifest of valid segments (path + sample offset).

**Online training** (`MoMFMADataset`, used by `pretrain_mixit.py`):
- the training input is **6 s** (not 10) — a 6 s window is randomly cropped from each valid 10 s segment;
- a MoM is `x1 + x2` from two different segments, preferring different tracks;
- stereo (M=2), RMS-normalized; the loss is the negative thresholded SNR (τ=1e-3, 30 dB clamp).

Run it:
```bash
# Stage 0: build the manifest once (needs torchaudio + the FMA audio)
python scripts/build_fma_manifest.py --clip_dir /path/to/fma_large \
    --out fma_manifest.json --silence_threshold 1e-4

# Stage 1: real pre-training from the manifest
python scripts/pretrain_mixit.py --manifest fma_manifest.json \
    --n_srcs 12 --n_layers 6 --emb_dim 128 --train_seconds 6 --batch_size 128
```

The one value to tune to your data is `--silence_threshold`: the paper defines silence via per-second power but does not give a numeric floor (FMA is MP3-sourced with reduced HF energy). Start at `1e-4` on RMS-scaled audio and inspect how many segments survive.

## MUSDB fine-tuning recipe (implemented)

The MUSDB stem loading now implements the paper's recipe, including the source-activity detection it inherits from BSRNN (Luo & Yu, Sec. IV-A1):

**Offline SAD** (`build_musdb_sad.py` → `sad.build_musdb_sad_manifest`): for each track and stem, split into 6 s segments with 50% overlap, split each segment into 10 chunks, compute chunk energies (silent chunks floored to 1e-5), set a per-track threshold = max(15%-quantile of all chunk energies, 1e-3), and keep a segment as salient iff more than 50% of its chunks exceed the threshold. Writes a manifest of salient per-stem offsets.

**Online dynamic mixing** (`MUSDBDataset` with `sad_manifest=`): each stem is sampled from a salient segment (different tracks per stem), RMS-normalized, scaled by a random gain in [-10, 10] dB, and dropped with probability 0.05 to simulate inactive sources. The four stems are summed to form the mixture.

Run it:
```bash
# Stage 0b: SAD manifest over the MUSDB training stems
python scripts/build_musdb_sad.py --musdb_root /path/to/musdb18hq/train \
    --out musdb_sad.json --segment_seconds 6

# Stage 2: fine-tune from the pre-trained model, using SAD-filtered stems
python scripts/finetune_musdb.py --musdb_root /path/to/musdb18hq/train \
    --sad_manifest musdb_sad.json --pretrained pretrained_mixit.pth \
    --n_srcs 12 --n_layers 6 --emb_dim 128
```

Without `--sad_manifest`, the dataset falls back to random-track random-crop sampling (still functional, just without silence filtering).

## What remains

- **Scale hyperparameters**: pre-train 150 epochs (~1000 steps each), fine-tune 900 epochs (~110 steps each), AdamW wd=1e-2, peak LR 1e-3 (5e-4 large), 5000-step linear warmup then 0.965/epoch decay, grad-clip L2=5, AMP + flash-attention. The YAML configs carry these; the real run needs GPUs (paper batch size 128 at 6 s for pre-training, 32 for fine-tuning).

## Notes on technical honesty

- `efficient_mixit_loss` uses a scale-relative ridge and falls back to `lstsq` when the gram matrix is singular (which happens when estimated sources are nearly collinear). It matches the exhaustive search on the constructed test, but is not guaranteed identical on real data; the efficient-vs-exhaustive test monitors this.
- `sparsity_loss` is necessary, not optional, when N ≫ K: without it, pre-training tends toward over-separation (e.g. drums split into kick/hi-hat across channels).
- Channel selection is a greedy per-stem realization based on SNR; the paper describes the procedure in 4 steps (Sec. 4.3), of which this is one practical instance.

## License and attribution

The `mixit_mss/bslocoformer/` code is Copyright (C) 2024 MERL, Apache-2.0 (license included in that folder), vendored unmodified from `merlresearch/tf-locoformer`. The MixIT layer around it is provided as scaffolding. Please cite:

- Saijo et al., *TF-Locoformer*, IWAENC 2024.
- Saijo et al., *Task-Aware Unified Source Separation*, ICASSP 2025 (band-split / BS-Locoformer).
- Saijo & Bando, *Exploring MixIT for Unsupervised Pre-training in Music Source Separation*.
