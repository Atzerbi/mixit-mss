# mixit-mss

**MixIT unsupervised pre-training for music source separation (MSS)**, built on top of MERL's **BS-Locoformer** (vendored into this repo). Reproduces the scheme of *"Is MixIT Really Unsuitable for Correlated Sources? Exploring MixIT for Unsupervised Pre-training in Music Source Separation"* (Saijo & Bando, arXiv:2505.07631):

> MixIT pre-training on the Free Music Archive (unlabeled) → channel selection → supervised fine-tuning on MUSDB18 → evaluation.

## What this repo is

The separator itself (BS-Locoformer) is MERL's, vendored **unmodified** under `mixit_mss/bslocoformer/` (Apache-2.0). This repo adds the **MixIT layer** around it: the efficient MixIT loss, the mixture-of-mixtures dataset, the exact FMA/SAD data recipes, channel selection, the two-stage training loop, inference, and evaluation.

The model works entirely in the **TF domain** (complex spectrogram in, complex spectrograms out); the (i)STFT is handled outside in `mixit_mss/stft.py`. It is stereo-native (`stereo=True`) and uses a Q=62 band-split encoder. `SeparatorAdapter` wraps STFT → BS-Locoformer → iSTFT so the MixIT loss compares waveforms.

**Why BS-Locoformer and not Demucs?** MixIT needs the model to emit **more outputs than real stems** (N=12 vs 4 VDBO); this over-provisioning is the mechanism that forces separation of the mixture-of-mixtures. BS-Locoformer exposes N (`num_spk`) as a hyperparameter, whereas a fixed-4-stem model like Demucs does not. The band-split also suits music, where sub-bands have very different statistics. Note that N=12 is only for pre-training: channel selection reduces it to the 4 VDBO stems for the final separator.

## Installation

```bash
pip install -r requirements.txt      # torch, torchaudio, rotary-embedding-torch, ...
pip install -e .                     # makes `import mixit_mss` work from anywhere
pip install museval                  # optional: only needed for the cSDR metric
```

## Repository layout

```
mixit_mss/
  bslocoformer/          # VENDORED real model (MERL, Apache-2.0), unmodified
  stft.py                # external STFT/iSTFT (the model is TF-domain only)
  separator_adapter.py   # STFT -> BS-Locoformer -> iSTFT; real + stub builders
  losses.py              # efficient MixIT loss (Eq.3) + exhaustive (for tests) + sparsity
  fma.py                 # FMA recipe: segmentation + silence rule + MoM loader (Sec. 4.1/4.3)
  sad.py                 # BSRNN source-activity detection (energy thresholding)
  datasets.py            # MoMDataset + MUSDBDataset (VDBO, SAD-aware)
  channel_selection.py   # maps the 12 MixIT channels onto the 4 VDBO stems (Sec. 4.3)
  pit.py                 # PIT / direct loss for fine-tuning
  inference.py           # chunked overlap-add separation, notebook-friendly
  evaluation.py          # uSDR (dependency-free) and cSDR (museval) metrics
scripts/
  build_fma_manifest.py  # Stage 0a: scan FMA, apply recipe, write manifest
  build_musdb_sad.py     # Stage 0b: SAD over MUSDB stems, write manifest
  pretrain_mixit.py      # Stage 1: MixIT pre-training (resumable)
  finetune_musdb.py      # Stage 2: channel selection + supervised fine-tuning
configs/                 # example YAML with the paper's recipe values
notebooks/
  SeparationDemo.ipynb   # load a model, separate, listen, save, evaluate
tests/
  test_pipeline.py       # shapes, backward, efficient-vs-exhaustive, real model, FMA, SAD
  test_inference.py      # overlap-add reconstruction + both inference scenarios
```

## Quick check (no data, no GPU needed)

```bash
PYTHONPATH=. python tests/test_pipeline.py     # includes a real-BS-Locoformer forward
PYTHONPATH=. python tests/test_inference.py    # needs torchaudio (writes temp wavs)
```

`--stub` on any script swaps in a lightweight TF-domain stub for fast shape/flow checks without building the large model.

## End-to-end pipeline: the full command sequence

The four stages, in order. Stages 0a/0b are one-off indexing passes; run them once.

### Stage 0a — FMA manifest (unlabeled pre-training data)

```bash
python scripts/build_fma_manifest.py \
    --clip_dir /path/to/fma_small \
    --out fma_manifest.json --silence_threshold 1e-4
```

Recipe (Sec. 4.1): 10 s segments with 5 s overlap; discard a segment with >5 s silence (power over 1 s intervals); drop audio <44.1 kHz, downsample >44.1 kHz. Writes a JSON of valid segments (path + offset — no audio is copied). Tune `--silence_threshold` to your data (FMA is MP3-sourced; start at `1e-4` and check how many segments survive). A few corrupt FMA files failing to open is expected and they are skipped.

### Stage 1 — MixIT pre-training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python scripts/pretrain_mixit.py \
    --manifest fma_manifest.json \
    --batch_size 1 --accum_steps 8 --train_seconds 6
```

Defaults reproduce the paper's medium-model schedule (Sec. 4.3): 150 epochs × 1000 steps, AdamW wd=1e-2, lr 0→1e-3 over 5000 steps then ×0.965/epoch, grad-clip L2=5, thresholded-SNR loss alone. A "step" is one optimiser update; `--accum_steps k` makes the effective batch `batch_size × k`, so `1 × 8` here. The paper's batch of 128 needs multiple GPUs; on one 24 GB GPU use `batch_size 1` and raise `--accum_steps` (8–16) toward it. Training loss prints every `--log_every` steps. There is deliberately **no validation loss**: pre-training is unsupervised and its quality is judged downstream by fine-tuned SDR, exactly as in the paper.

Produces `pretrained_mixit.pth` (weights) and `pretrained_mixit.pth.ckpt` (full training state, for `--resume`).

### Stage 0b — MUSDB SAD manifest (before fine-tuning)

```bash
python scripts/build_musdb_sad.py \
    --musdb_root /path/to/musdb18hq/train \
    --out musdb_sad.json --segment_seconds 6
```

Energy-based source-activity detection from BSRNN (Sec. IV-A1): per track and stem, 6 s segments at 50% overlap, 10 chunks each, threshold = max(15%-quantile of chunk energies, 1e-3), keep a segment if >50% of its chunks exceed it. Removes silent regions from the training stems.

### Stage 2 — Supervised fine-tuning (channel selection + training)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python scripts/finetune_musdb.py \
    --musdb_root /path/to/musdb18hq/train \
    --sad_manifest musdb_sad.json \
    --pretrained pretrained_mixit.pth \
    --batch_size 1 --accum_steps 8
```

On start it runs **channel selection** (maps the 12 pre-trained channels to the 4 VDBO stems on the validation set; prints `stem->channel map: {...}`), then fine-tunes with supervision. Dynamic mixing per the paper: each stem sampled from a salient SAD segment, RMS-normalized, random gain [-10,10] dB, dropped with p=0.05. Without `--sad_manifest` it falls back to random-crop sampling. Saves `finetuned_musdb.pth` (weights **plus** the channel map), which inference and evaluation consume.

To reproduce the paper's central claim, also fine-tune **from scratch** (omit `--pretrained`) and compare — the pre-training vs from-scratch gap is the result.

### Stage 3 — Inference and evaluation

Open `notebooks/SeparationDemo.ipynb`, or from Python:

```python
from mixit_mss.inference import load_separator, separate_musdb_test
from mixit_mss.evaluation import evaluate_musdb_test, summarize

sep = load_separator("finetuned_musdb.pth", device="cuda")   # auto-detects VDBO mode
separate_musdb_test(sep, "/path/to/musdb18hq/test", "sep_out", limit=3)   # write wavs

df = evaluate_musdb_test(sep, "/path/to/musdb18hq/test", metrics=("usdr", "csdr"))
print(summarize(df).round(2))    # per-stem + Average, like the paper's Table 1
```

`load_separator` reads the model hyperparameters you pass (`n_srcs`, `n_layers`, `emb_dim`, `stft_size`, ...) — they **must** match training. Evaluation needs a fine-tuned model; a raw pre-training model has unlabeled channels that cannot be scored against VDBO.

## Resuming an interrupted run

Both `pretrain_mixit.py` and `finetune_musdb.py` are resumable with the same mechanism. Each save writes two files atomically (temp + rename, so a job killed mid-write never corrupts the previous checkpoint):

| file | contents | consumed by |
|---|---|---|
| `--out` | weights (pre-train) / weights + channel map (fine-tune) | inference, fine-tuning, notebook |
| `--ckpt_out` (default `<out>.ckpt`) | weights + optimizer state + step (+ lr schedule for pre-train, + channel map for fine-tune) | `--resume` |

Continue exactly where it stopped:

```bash
# pre-training
PYTHONPATH=. python scripts/pretrain_mixit.py --manifest fma_manifest.json \
    --batch_size 1 --accum_steps 8 --resume

# fine-tuning
PYTHONPATH=. python scripts/finetune_musdb.py --musdb_root .../train \
    --sad_manifest musdb_sad.json --batch_size 1 --accum_steps 8 --resume
```

Pass the **same** hyperparameters as the original run: weights load strictly, so a changed `--n_layers`/`--emb_dim` fails loudly instead of silently training a different model. For fine-tuning, the **channel map is restored from the checkpoint**, so channel selection is *not* re-run on resume — you get the same 12→4 mapping the interrupted run used, and no wasted compute. A weights-only checkpoint from before `.ckpt` existed can still be continued, but tell it where it stopped so state is re-placed rather than restarting: `--resume <weights>.pth --start_step 25000` (pre-train re-places the lr schedule; fine-tune resets the optimiser moments).

Launch detached so a dropped SSH connection (SIGHUP) doesn't kill the run:

```bash
tmux new -s mixit        # run inside, detach with Ctrl-b d
# or:
nohup PYTHONPATH=. python scripts/pretrain_mixit.py ... > train.log 2>&1 &
```

## Model configuration (medium, paper Sec. 4.2)

`build_bslocoformer_adapter()` defaults reproduce the paper's medium model: `n_srcs=12`, `n_layers=6` (B), `emb_dim=128` (D), `ffn_hidden_dim=384` (C), `conv1d_kernel=8` (K), `n_heads=8` (H), `num_groups=8` (G), `stft_size=2048`, `hop_length=512`. This gives a **15.0M-parameter separator**, matching the paper. (The paper text lists C=192, but that yields 7.9M, inconsistent with its own stated 15.0M; C=384 reproduces it.) The full model is larger (~79M) because the band-split decoder scales with N=12 outputs — this is inherent to MixIT, not a bug. `stft_size` must equal the external STFT `n_fft`.

## Notes on fidelity and known caveats

- **No sparsity loss in pre-training.** The paper (Sec. 2.3) uses the thresholded-SNR loss alone and leaves over-separation to be resolved by fine-tuning, so `--lambda_sparse` defaults to 0. A `sparsity_loss` is provided for experimentation only; using it deviates from the paper.
- **STFT window** (2048/512) is not stated for music in the paper; 2048 @ 44.1 kHz is the standard MSS choice, used here as a reasonable default.
- **`efficient_mixit_loss`** uses a scale-relative ridge and falls back to `lstsq` on singular gram matrices (near-collinear estimates). It matches the exhaustive O(2^N) search on the constructed test; `test_pipeline.py` monitors this.
- **Channel selection** is a greedy per-stem realization based on SNR — one practical instance of the paper's 4-step procedure (Sec. 4.3).

## License and attribution

`mixit_mss/bslocoformer/` is Copyright (C) 2024 MERL, Apache-2.0 (license in that folder), vendored unmodified from [`merlresearch/tf-locoformer`](https://github.com/merlresearch/tf-locoformer). Please cite:

- Saijo & Bando, *Is MixIT Really Unsuitable for Correlated Sources? Exploring MixIT for Unsupervised Pre-training in Music Source Separation*, 2025.
- Saijo et al., *Task-Aware Unified Source Separation*, ICASSP 2025 (band-split / BS-Locoformer).
- Saijo et al., *TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation and Enhancement*, IWAENC 2024.
- Wisdom et al., *Unsupervised Sound Separation Using Mixture Invariant Training*, NeurIPS 2020 (MixIT); and *Sparse, Efficient, and Semantic MixIT*, WASPAA 2021 (efficient solver).
