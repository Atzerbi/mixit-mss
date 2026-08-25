"""Permutation Invariant Training (PIT) for supervised fine-tuning.

In fine-tuning the model emits K=4 channels (after channel selection) to be aligned
with the 4 VDBO stems. If the order is fixed (channels already selected per stem) a
direct loss can be used; PIT remains useful if one prefers not to constrain the order.
"""

import itertools
import torch
from .losses import negative_snr

def pit_loss(est, targets, loss_fn=negative_snr):
    """est, targets: [B, K, C, L]. Returns (loss, best_perm[B, K])."""
    B, K, C, L = est.shape
    perms = list(itertools.permutations(range(K)))
    best = None
    best_perm = None
    for perm in perms:
        e = est[:, list(perm)]                       # [B,K,C,L]
        l = loss_fn(e, targets).mean(dim=(1, 2))     # [B]
        if best is None:
            best, best_perm = l, torch.tensor(perm, device=est.device).expand(B, K)
        else:
            imp = l < best
            best = torch.where(imp, l, best)
            p = torch.tensor(perm, device=est.device).expand(B, K)
            best_perm = torch.where(imp[:, None], p, best_perm)
    return best.mean(), best_perm

def direct_loss(est, targets, loss_fn=negative_snr):
    """Direct loss when the channel order is already fixed (post channel-selection).
    est, targets: [B, K, C, L]."""
    return loss_fn(est, targets).mean()
