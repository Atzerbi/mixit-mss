"""Channel selection after MixIT pre-training.

The pre-trained model produces N unordered, unlabeled outputs (e.g. N=12). For
supervised fine-tuning we need to map N -> 4 VDBO stems. Procedure (Sec. 4.3 of
the Saijo & Bando paper):

  1. on a validation set with VDBO ground truth, separate each mixture -> N estimates;
  2. for each example, find the optimal permutation/assignment aligning a subset
     of the N estimates to the 4 targets (maximize SNR);
  3. accumulate, for each stem, which channel gets chosen most often;
  4. select for each stem the most frequently aligned channel.

Here we implement the greedy per-stem selection based on mean SNR, which is a
practical realization of steps 2-4. It returns a list of 4 indices (one per stem)
to read the model's outputs during fine-tuning.
"""

import torch
from collections import Counter
from .losses import negative_snr

STEMS = ("vocals", "drums", "bass", "other")


@torch.no_grad()
def _best_channel_per_stem(est, targets):
    """For a batch, find for each stem the channel (out of N) with the best SNR.
    est     : [B, N, C, L]
    targets : [B, K, C, L]  (K=4)
    returns : LongTensor [B, K] with the channel index chosen for each stem.
    """
    B, N, C, L = est.shape
    K = targets.shape[1]
    # SNR of each channel against each stem: [B, K, N]
    est_e = est.unsqueeze(1)          # [B,1,N,C,L]
    tgt_e = targets.unsqueeze(2)      # [B,K,1,C,L]
    snr = -negative_snr(est_e, tgt_e).mean(dim=-1)   # average over channels -> [B,K,N]
    best = snr.argmax(dim=-1)         # [B,K]
    return best


@torch.no_grad()
def select_channels(model, val_loader, n_srcs, device="cpu", max_batches=None):
    """Determine the stem->channel map on the validation set.

    model      : SeparatorAdapter (or compatible) mapping mix [B,C,L] -> [B,N,C,L].
    val_loader : yields (mix [B,C,L], targets [B,4,C,L]).
    n_srcs     : N.
    returns    : dict {stem_name: channel_index} and the counts matrix [K, N].
    """
    model.eval()
    K = len(STEMS)
    counts = torch.zeros(K, n_srcs, dtype=torch.long)

    for bi, (mix, targets) in enumerate(val_loader):
        if max_batches is not None and bi >= max_batches:
            break
        mix, targets = mix.to(device), targets.to(device)
        est = model(mix)                       # [B, N, C, L]
        best = _best_channel_per_stem(est, targets)   # [B, K]
        for k in range(K):
            for ch in best[:, k].tolist():
                counts[k, ch] += 1

    # greedy assignment: most "decisive" stem first, avoids channel collisions
    mapping = {}
    used = set()
    order = counts.max(dim=1).values.argsort(descending=True).tolist()
    for k in order:
        ranked = counts[k].argsort(descending=True).tolist()
        for ch in ranked:
            if ch not in used:
                mapping[STEMS[k]] = ch
                used.add(ch)
                break
    return mapping, counts
