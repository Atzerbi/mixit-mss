"""MixIT loss and associated components.

Contains:
  - negative_snr          : reconstruction loss (SNR with soft-threshold)
  - exhaustive_mixit_loss : exhaustive O(M^N) search — ONLY for validation with small N
  - efficient_mixit_loss  : least-squares solver + binarization (Eq. 3, Wisdom 2021)
  - sparsity_loss         : anti-over-separation penalty

Shape convention (stereo-aware):
  est_sources : [B, N, C, L]   N separator outputs, C channels, L samples
  mixtures    : [B, M, C, L]   M mixtures of the MoM (typically M=2)
"""

import itertools
import torch
import torch.nn.functional as F


def negative_snr(est, ref, eps=1e-3):
    """Negative SNR with soft-threshold ~30 dB (stable when ref is near-silence).
    est, ref: [..., L]. Reduces over the last axis (time) and returns [...]."""
    err = ref - est
    num = (ref ** 2).sum(dim=-1)
    den = (err ** 2).sum(dim=-1) + eps * num
    return -10 * torch.log10(num / (den + 1e-8) + 1e-8)


def _apply_assignment(est_sources, A):
    """Remix: sum the sources assigned to each mixture.
    est_sources [B,N,C,L], A [B,M,N] -> est_mix [B,M,C,L]."""
    return torch.einsum("bmn,bncl->bmcl", A, est_sources)


def exhaustive_mixit_loss(est_sources, mixtures, loss_fn=negative_snr):
    """Original MixIT (Wisdom 2020): tries ALL source->mixture assignments.
    Cost O(M^N). Use ONLY to validate efficient_mixit_loss with N<=~6.
    Returns (loss, A_best [B,M,N])."""
    B, N, C, L = est_sources.shape
    M = mixtures.shape[1]

    # all assignments: each source goes to one of the M mixtures
    all_assign = list(itertools.product(range(M), repeat=N))  # M^N tuples
    best_loss = None
    best_A = None
    for assign in all_assign:
        A = torch.zeros(B, M, N, device=est_sources.device)
        for n, m in enumerate(assign):
            A[:, m, n] = 1.0
        est_mix = _apply_assignment(est_sources, A)          # [B,M,C,L]
        # loss averaged over mixtures and channels
        l = loss_fn(est_mix, mixtures).mean(dim=(1, 2))       # [B]
        if best_loss is None:
            best_loss, best_A = l, A
        else:
            improved = l < best_loss
            best_loss = torch.where(improved, l, best_loss)
            best_A = torch.where(improved[:, None, None], A, best_A)
    return best_loss.mean(), best_A


def efficient_mixit_loss(est_sources, mixtures, loss_fn=negative_snr, ridge=1e-4):
    """Efficient MixIT: solves the continuous mixing matrix via least squares
    and projects it onto the binary constraint (each source to a single mixture).
    Approximates exhaustive_mixit_loss at O(N^3) instead of O(M^N).

    est_sources [B,N,C,L], mixtures [B,M,C,L] -> (loss, A [B,M,N])."""
    B, N, C, L = est_sources.shape
    M = mixtures.shape[1]

    est_flat = est_sources.reshape(B, N, C * L)                     # [B,N,CL]
    mix_flat = mixtures.reshape(B, M, C * L)                         # [B,M,CL]

    # A_cont = mix est^T (est est^T + ridge I)^-1     -> [B,M,N]
    gram = torch.bmm(est_flat, est_flat.transpose(1, 2))            # [B,N,N]
    cross = torch.bmm(mix_flat, est_flat.transpose(1, 2))           # [B,M,N]

    # scale-relative ridge (robust when sources are nearly collinear and the
    # gram matrix is ill-conditioned / singular)
    diag_mean = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-8)  # [B]
    eye = torch.eye(N, device=est_flat.device).unsqueeze(0)
    gram = gram + (ridge * diag_mean).view(B, 1, 1) * eye
    try:
        A_cont = torch.linalg.solve(gram, cross.transpose(1, 2)).transpose(1, 2)
    except torch.linalg.LinAlgError:
        # fallback: least squares (handles still-singular gram)
        A_cont = torch.linalg.lstsq(gram, cross.transpose(1, 2)).solution.transpose(1, 2)

    # binary projection P_B: each source (column) to the mixture with max weight
    assign = A_cont.argmax(dim=1)                                   # [B,N]
    A = F.one_hot(assign, num_classes=M).permute(0, 2, 1).float()   # [B,M,N]

    est_mix = _apply_assignment(est_sources, A)                     # [B,M,C,L]
    loss = loss_fn(est_mix, mixtures).mean()
    return loss, A


def sparsity_loss(est_sources, gamma=1.0):
    """Penalizes activating too many sources (L1/L2 ratio over energies).
    Crucial when N (outputs) >> K (real sources), as in MixIT pre-training.
    est_sources: [B,N,C,L] -> scalar."""
    power = (est_sources ** 2).sum(dim=(-1, -2))   # [B,N] energy per source
    l1 = power.sqrt().sum(dim=1)                    # [B]
    l2 = power.sum(dim=1).sqrt()                    # [B]
    return (gamma * l1 / (l2 + 1e-8)).mean()
