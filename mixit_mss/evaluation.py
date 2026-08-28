"""Evaluation metrics for music source separation.

Implements the two metrics reported in Saijo & Bando (arXiv:2505.07631, Sec. 4.3):

  uSDR (utterance/song-level SDR)
      The song-level signal-to-distortion ratio defined as a global SNR over the
      whole track: 10*log10(||ref||^2 / ||ref - est||^2), computed per stem over
      the entire song and then averaged. This is the "uSDR" of the Music Demixing
      Challenge (Mitsufuji et al.), robust and simple.

  cSDR (chunk-wise SDR)
      The museval/BSSEval v4 SDR, computed on 1-second frames and aggregated with
      the median (the SiSEC/MDX convention). Requires the `museval` package. This
      is the stricter, more standard MSS metric.

Both are computed against the MUSDB ground-truth stems. uSDR needs no extra
dependency; cSDR needs `pip install museval`.

Typical use:
    from mixit_mss.evaluation import evaluate_musdb_test
    df = evaluate_musdb_test(sep, "/path/musdb18hq/test", metrics=("usdr", "csdr"))
    print(df)                      # per-track and per-stem table
    print(df.groupby("stem").mean(numeric_only=True))   # averages like the paper
"""

import os
import numpy as np
import torch

from .inference import separate, _load_audio
from .channel_selection import STEMS


# ---------------------------------------------------------------------------
# uSDR — song-level global SDR (no extra dependency)
# ---------------------------------------------------------------------------
def usdr(ref, est, eps=1e-7):
    """Song-level SDR for one stem.
    ref, est: np.ndarray [C, L] (or [L]). Returns a float in dB.
    Definition: 10*log10(||ref||^2 / ||ref - est||^2), matching MDX uSDR.
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    num = (ref ** 2).sum()
    den = ((ref - est) ** 2).sum()
    return 10.0 * np.log10((num + eps) / (den + eps))


# ---------------------------------------------------------------------------
# cSDR — museval BSSEval v4 (median over 1 s frames)
# ---------------------------------------------------------------------------
def csdr_museval(refs, ests, sr=44100):
    """Chunk-wise SDR via museval for all stems at once.
    refs, ests: dict {stem: np.ndarray [L, C]} (museval wants time-major).
    Returns dict {stem: median cSDR in dB}. Requires `museval`.
    """
    import museval
    # museval.evaluate expects arrays shaped (nsrc, nsampl, nchan)
    names = list(refs.keys())
    ref_arr = np.stack([refs[n] for n in names], axis=0)   # [S, L, C]
    est_arr = np.stack([ests[n] for n in names], axis=0)   # [S, L, C]
    sdr, isr, sir, sar = museval.evaluate(ref_arr, est_arr, win=sr, hop=sr)
    # sdr: [S, nframes]; median over frames ignoring NaNs (museval convention)
    out = {}
    for i, n in enumerate(names):
        out[n] = float(np.nanmedian(sdr[i]))
    return out


# ---------------------------------------------------------------------------
# Whole-test-split evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_musdb_test(sep, musdb_test_root, metrics=("usdr",), limit=None,
                        chunk_seconds=12.0, overlap=0.5, mixture_name="mixture.wav",
                        channel_map=None, verbose=True):
    """Separate every MUSDB test track and score it against the ground-truth stems.

    sep         : a Separator. Normally fine-tuned (VDBO). For evaluating a
                  pre-training-only model, pass an explicit `channel_map`
                  (see channel_selection.select_channels) to map its raw channels
                  to VDBO stems without fine-tuning.
    metrics     : any of ("usdr", "csdr"). csdr needs the museval package.
    channel_map : optional {stem: channel_index}. If given, it overrides the
                  separator's own map for this evaluation (used for the
                  channel-selection-only diagnostic).
    Returns     : a pandas DataFrame with columns [track, stem, <metric>...].

    The paper reports averages over stems; do df.groupby("stem").mean() to match.
    """
    import pandas as pd

    # Resolve the channel map to use: explicit arg wins, else the separator's own.
    eff_map = channel_map if channel_map is not None else sep.channel_map
    if eff_map is None:
        raise ValueError(
            "No channel map available. Either load a fine-tuned model, or pass "
            "channel_map= from channel_selection.select_channels to evaluate a "
            "pre-training-only model.")
    # Temporarily install the map so `separate()` returns named VDBO stems.
    saved_map = sep.channel_map
    sep.channel_map = eff_map
    try:
        metrics = tuple(m.lower() for m in metrics)
        tracks = sorted(d for d in os.listdir(musdb_test_root)
                        if os.path.isdir(os.path.join(musdb_test_root, d)))
        if limit is not None:
            tracks = tracks[:limit]

        rows = []
        for ti, track in enumerate(tracks):
            tdir = os.path.join(musdb_test_root, track)
            mix_path = os.path.join(tdir, mixture_name)
            if not os.path.exists(mix_path):
                if verbose:
                    print(f"[skip] {track}: no {mixture_name}", flush=True)
                continue

            # separate
            est = separate(sep, mix_path, chunk_seconds=chunk_seconds, overlap=overlap)

            # load ground-truth stems, aligned in length to the estimates
            refs, ests = {}, {}
            for stem in STEMS:
                gt_path = os.path.join(tdir, f"{stem}.wav")
                if not os.path.exists(gt_path):
                    continue
                ref = _load_audio(gt_path, sep.sr)                 # [C, L]
                e = est[stem]                                       # [C, L]
                L = min(ref.shape[-1], e.shape[-1])
                refs[stem] = ref[..., :L].cpu().numpy()
                ests[stem] = e[..., :L].cpu().numpy()

            # metrics
            csdr_vals = {}
            if "csdr" in metrics:
                refs_tc = {n: refs[n].T for n in refs}             # -> [L, C]
                ests_tc = {n: ests[n].T for n in ests}
                csdr_vals = csdr_museval(refs_tc, ests_tc, sr=sep.sr)

            for stem in refs:
                row = {"track": track, "stem": stem}
                if "usdr" in metrics:
                    row["uSDR"] = usdr(refs[stem], ests[stem])
                if "csdr" in metrics:
                    row["cSDR"] = csdr_vals.get(stem, float("nan"))
                rows.append(row)

            if verbose:
                msg = f"[{ti+1}/{len(tracks)}] {track}"
                if "usdr" in metrics:
                    mu = np.mean([r["uSDR"] for r in rows if r["track"] == track])
                    msg += f" | uSDR {mu:.2f}"
                if "csdr" in metrics:
                    mc = np.nanmean([r["cSDR"] for r in rows if r["track"] == track])
                    msg += f" | cSDR {mc:.2f}"
                print(msg, flush=True)

        return pd.DataFrame(rows)
    finally:
        sep.channel_map = saved_map    # restore, even if evaluation raised


def summarize(df):
    """Per-stem averages plus the overall mean, in the style of the paper's tables.
    Returns a DataFrame indexed by stem with an added 'Average' row.
    """
    import pandas as pd
    num = df.groupby("stem").mean(numeric_only=True)
    # reorder stems as VDBO and append the average
    num = num.reindex([s for s in STEMS if s in num.index])
    avg = num.mean(axis=0).to_frame().T
    avg.index = ["Average"]
    return pd.concat([num, avg])


@torch.no_grad()
def evaluate_pretrained_with_selection(
        sep, musdb_train_root, musdb_test_root, n_srcs=12,
        metrics=("usdr",), sel_max_batches=8, sel_segment_len=None,
        limit=None, chunk_seconds=12.0, overlap=0.5, sad_manifest=None,
        device=None, verbose=True):
    """Channel-selection-only diagnostic (no fine-tuning).

    Reproduces the 3-step selection of the paper (Sec. 4.3) on the MUSDB *validation*
    tracks, then evaluates the pre-trained model as-is on the *test* set. This
    measures what MixIT pre-training learned before any supervised fine-tuning.

    Selection is done on the training/validation split and evaluation on the test
    split, so the map is not tuned on the data it is scored on.

    metrics: any of ("usdr", "csdr") - the same metrics used elsewhere; csdr needs museval.
    Returns (df, channel_map).
    """
    from torch.utils.data import DataLoader
    from .datasets import MUSDBDataset
    from .channel_selection import select_channels

    device = device or sep.device
    seg = sel_segment_len or int(round(6.0 * sep.sr))   # paper: 6 s chunks for selection

    # --- Step 1-3: channel selection on the validation (train-split) tracks ------
    val_ds = MUSDBDataset(root=musdb_train_root, segment_len=seg,
                          n_channels=sep.n_channels, dynamic_mixing=False,
                          sad_manifest=sad_manifest)
    val_dl = DataLoader(val_ds, batch_size=1, drop_last=True)
    if verbose:
        print("running channel selection on the validation split...", flush=True)
    mapping, counts = select_channels(sep.model, val_dl, n_srcs=n_srcs,
                                      device=device, max_batches=sel_max_batches)
    if verbose:
        print("stem->channel map:", mapping, flush=True)

    # --- Evaluate the pre-trained model on the test split with that map ----------
    df = evaluate_musdb_test(sep, musdb_test_root, metrics=metrics, limit=limit,
                             chunk_seconds=chunk_seconds, overlap=overlap,
                             channel_map=mapping, verbose=verbose)
    return df, mapping