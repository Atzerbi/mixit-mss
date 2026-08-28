"""Stage 1 - MixIT pre-training on unlabeled data (FMA).

MixIT (Mixture-Invariant Training) trains a separator without labels: it feeds the
model the sum of two random mixtures (a "mixture of mixtures", MoM) and asks it to
split that sum back into the two original mixtures. The model learns to separate
sources as a side effect, without ever seeing isolated stems.

This script runs the pre-training described in [1]. The separator is the BS-Locoformer
(a time-frequency model that works on spectrograms) [2]. After pre-training, use
finetune_musdb.py to map the outputs to the 4 VDBO stems and fine-tune supervised.

Usage (paper recipe, Saijo & Bando, arXiv:2505.07631 Sec. 4.3):
    python scripts/pretrain_mixit.py --manifest fma_manifest.json

The defaults reproduce the paper's medium-model schedule:
    150 epochs x 1000 steps          --epochs / --steps_per_epoch
    effective batch 128              --batch_size x --accum_steps
    AdamW, weight decay 1e-2         --weight_decay
    lr 0 -> 1e-3 over 5000 steps,    --lr / --warmup_steps
      then x0.965 at each epoch end  --lr_decay
    gradient L2-norm clipped to 5    --clip_grad
    thresholded-SNR loss only        --lambda_sparse 0 (paper uses no sparsity here)

A "step" is one *optimizer update*: with --accum_steps k each step consumes k
micro-batches, so the effective batch is batch_size * k. That is what makes the
paper's batch of 128 reachable on a single GPU.

Checkpointing / resuming
    Every save writes TWO files, atomically (temp file + rename, so a job killed
    mid-write never leaves a truncated checkpoint):
      --out       weights only, what inference.py / finetune_musdb.py consume;
      --ckpt_out  full training state (weights + AdamW moments + lr schedule +
                  step counter), the file to resume from.
    To continue an interrupted run:
        python scripts/pretrain_mixit.py --manifest fma_manifest.json --resume
    An older run that only saved weights can still be continued by declaring where
    it stopped, which re-places the lr schedule (the AdamW moments are gone):
        ... --resume pretrained_mixit.pth --start_step 25000
    To recover the hyperparameters of an interrupted run, this code snippet prints
    them for you:
        import torch
        ck = torch.load("pretrained_mixit.pth.ckpt", map_location="cpu", weights_only=True)
        print("step:", ck["step"])
        for k in ("batch_size", "accum_steps", "lr", "warmup_steps", "lr_decay",
          "weight_decay", "clip_grad", "steps_per_epoch", "epochs", "steps",
          "n_srcs", "n_layers", "emb_dim", "stft_size", "hop_length",
          "train_seconds", "lambda_sparse", "manifest"):
        print(f"  {k:16s} {ck['args'][k]}")
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from mixit_mss.datasets import MoMDataset
from mixit_mss.fma import MoMFMADataset
from mixit_mss.losses import efficient_mixit_loss, sparsity_loss
from mixit_mss.separator_adapter import build_stub_adapter, build_bslocoformer_adapter

def build_model(args, device):
    """Builds the separator.

    Default: REAL BS-Locoformer (vendored from MERL in mixit_mss/bslocoformer).
    --stub : lightweight TF-domain stub to validate the pipeline without the big model.
    """
    if args.stub:
        model = build_stub_adapter(n_srcs=args.n_srcs, n_channels=args.channels,
                                   stereo=args.stereo, n_fft=args.stft_size,
                                   hop_length=args.hop_length)
    else:
        model = build_bslocoformer_adapter(
            n_srcs=args.n_srcs, n_channels=args.channels, stereo=args.stereo,
            n_layers=args.n_layers, emb_dim=args.emb_dim,
            sample_rate=args.sr, stft_size=args.stft_size, hop_length=args.hop_length)
    return model.to(device)


def lr_lambda(warmup_steps, steps_per_epoch, decay):
    """Paper schedule: linear warmup to the peak lr, then a per-epoch decay.

    Returns the multiplier applied to the peak lr at optimiser step `step`. The two
    factors compose, so the decay keeps ticking during warmup and the curve has no
    discontinuity where the ramp ends.
    """
    def fn(step):
        warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        return warm * (decay ** (step // steps_per_epoch))
    return fn

def atomic_save(obj, path):
    """torch.save via a temporary file + rename.

    A checkpoint is written every save_every steps and takes seconds for a model
    this size; a process killed inside that window would otherwise destroy the only
    copy of the run. os.replace is atomic on the same filesystem, so the previous
    checkpoint stays intact until the new one is complete.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_all(model, opt, sched, step, args):
    """Weights (for inference/fine-tuning) + full training state (for --resume)."""
    atomic_save(model.state_dict(), args.out)
    atomic_save({"state_dict": model.state_dict(),
                 "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(),
                 "step": step,
                 "args": vars(args)}, args.ckpt_out)

def build_scheduler(opt, args, start_step):
    """LambdaLR placed directly at `start_step` instead of stepped there in a loop."""
    lam = lr_lambda(args.warmup_steps, args.steps_per_epoch, args.lr_decay)
    if start_step <= 0:
        return torch.optim.lr_scheduler.LambdaLR(opt, lam)
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])   # required by LambdaLR when last_epoch != -1
    # __init__ steps once, so last_epoch lands on start_step: the same lr the
    # interrupted run was using when it stopped.
    return torch.optim.lr_scheduler.LambdaLR(opt, lam, last_epoch=start_step - 1)

def main():
    ap = argparse.ArgumentParser()

    # --- DATA -----------------------------------------------------------------
    # JSON manifest built by build_fma_manifest.py (the paper-recipe FMA loader).
    ap.add_argument("--manifest", default=None, help="FMA manifest JSON (paper recipe)")
    # Legacy raw-folder path (no segmentation/silence recipe). Not needed with a manifest.
    ap.add_argument("--clip_dir", default=None, help="raw folder (legacy MoMDataset)")
    # Audio window length the model trains on. Paper: 6 s cropped from each 10 s segment.
    ap.add_argument("--train_seconds", type=float, default=6.0, help="training input length (paper: 6 s)")
    ap.add_argument("--use_fma", action="store_true", help="test only: exercise the FMA loader path")
    # Test-only: random noise instead of real audio (validates shapes/flow).
    ap.add_argument("--synthetic", action="store_true", help="test only: use random noise")
    # Use the lightweight stub model instead of the real BS-Locoformer.
    ap.add_argument("--stub", action="store_true", help="use the TF-domain stub, not the real model")

    # --- MODEL: BS-Locoformer MEDIUM (paper Sec. 4.2) -------------------------
    # N output sources. MixIT emits MORE channels (12) than real stems (4); channel
    # selection later maps 12 -> 4 (VDBO). Keep at 12.
    ap.add_argument("--n_srcs", type=int, default=12, help="model output sources (paper: 12)")
    ap.add_argument("--channels", type=int, default=2, help="audio channels (2 = stereo)")
    ap.add_argument("--stereo", action="store_true", default=True)
    ap.add_argument("--mono", dest="stereo", action="store_false")
    # Depth (B) and width (D) of the separator. Paper medium: B=6, D=128.
    ap.add_argument("--n_layers", type=int, default=6, help="Locoformer blocks B (paper: 6)")
    ap.add_argument("--emb_dim", type=int, default=128, help="embedding dim D (paper: 128)")

    # --- STFT -----------------------------------------------------------------
    ap.add_argument("--sr", type=int, default=44100, help="sample rate; must match the data")
    # FFT window; must match the value passed to the model. ~46 ms @ 44.1 kHz.
    ap.add_argument("--stft_size", type=int, default=2048, help="FFT window size (must match model)")
    ap.add_argument("--hop_length", type=int, default=512, help="STFT hop (window advance)")

    # --- Training length (paper counts optimiser steps in epochs) -------------
    ap.add_argument("--epochs", type=int, default=150, help="paper: 150")
    ap.add_argument("--steps_per_epoch", type=int, default=1000, help="paper: ~1000")
    ap.add_argument("--steps", type=int, default=0, help="total optimiser steps; 0 = epochs*steps_per_epoch")
    # (Legacy) sample length for the non-manifest MoMDataset path.
    ap.add_argument("--segment_len", type=int, default=32000, help="legacy MoMDataset segment length")

    # --- Optimiser / schedule (paper Sec. 4.3) --------------------------------
    # Micro-batches per weight update. Effective batch = batch_size * accum_steps.
    # On one 24 GB GPU: batch_size 1, accum_steps 8-16, train_seconds 3-6.
    ap.add_argument("--batch_size", type=int, default=1, help="micro-batch size (raise until OOM)")
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="micro-batches per optimiser step; effective batch = batch_size*accum_steps (paper: 128)")
    ap.add_argument("--lr", type=float, default=1e-3, help="peak lr (paper: 1e-3; 5e-4 for large)")
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay (paper: 1e-2)")
    ap.add_argument("--warmup_steps", type=int, default=5000, help="linear 0->lr steps (paper: 5000)")
    ap.add_argument("--lr_decay", type=float, default=0.965, help="per-epoch lr decay (paper: 0.965)")
    ap.add_argument("--clip_grad", type=float, default=5.0, help="max gradient L2-norm (paper: 5; 0 = off)")
    # The paper's pre-training loss is the thresholded SNR ALONE. Sparsity is left to
    # fine-tuning, so the default is 0. Set >0 only to experiment (deviates from paper).
    ap.add_argument("--lambda_sparse", type=float, default=0.0,
                    help="anti-over-separation penalty weight; paper uses 0 in pre-training")

    # --- Logging / checkpointing ---------------------------------------------
    ap.add_argument("--log_every", type=int, default=50, help="progress line every N optimiser steps")
    ap.add_argument("--save_every", type=int, default=0, help="checkpoint every N steps; 0 = once per epoch")
    ap.add_argument("--out", default="pretrained_mixit.pth", help="output weights file")
    ap.add_argument("--ckpt_out", default=None,
                    help="full training-state file for --resume (default: <out>.ckpt)")

    # --- Resuming an interrupted run -----------------------------------------
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="continue training; bare --resume picks up <out>.ckpt")
    ap.add_argument("--start_step", type=int, default=0,
                    help="step reached, ONLY needed when resuming from a weights-only file")
    args = ap.parse_args()
    if args.ckpt_out is None:
        args.ckpt_out = args.out + ".ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset: manifest -> FMA loader; else legacy folder loader.
    if args.manifest is not None or (args.synthetic and args.use_fma):
        ds = MoMFMADataset(manifest_path=args.manifest,
                           train_input_seconds=args.train_seconds,
                           _synthetic=args.synthetic)
    else:
        ds = MoMDataset(clip_dir=args.clip_dir, segment_len=args.segment_len,
                        n_channels=args.channels, _synthetic=args.synthetic)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Build model (BSLocoformer)
    model = build_model(args, device)

    total_steps = args.steps or args.epochs * args.steps_per_epoch
    save_every = args.save_every or args.steps_per_epoch

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Resume ---------------------------------------------------------------
    # The starting step has to be known BEFORE the scheduler exists, so the lr
    # curve can be rebuilt at the right position in one shot.
    ckpt, start_step = None, 0
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = args.ckpt_out if os.path.exists(args.ckpt_out) else args.out
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise SystemExit(f"--resume: no such checkpoint: {resume_path}")
        # weights_only=True: everything we save is tensors + primitives, so the
        # safe loader suffices and the file needs no trust.
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            start_step = int(ckpt["step"])          # full state: the run knows where it was
        else:
            # Weights-only file (older runs, or --out). AdamW's moments are gone;
            # the lr schedule still has to be re-placed by hand.
            start_step = args.start_step
            if start_step <= 0:
                print("WARNING: resuming from a weights-only checkpoint without "
                      "--start_step: the lr schedule restarts from warmup, which is "
                      "NOT what an interrupted run should do.", flush=True)

    sched = build_scheduler(opt, args, start_step)

    if ckpt is not None:
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(sd)                   # strict: catches mismatched hyperparams
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            print(f"resume     : {resume_path} -> step {start_step} "
                  f"(weights + AdamW moments + lr schedule)", flush=True)
        else:
            print(f"resume     : {resume_path} -> step {start_step} "
                  f"(weights + lr schedule; AdamW moments lost, expect a short "
                  f"loss bump while they rebuild)", flush=True)
        del ckpt

    if start_step >= total_steps:
        raise SystemExit(f"nothing to do: resumed at step {start_step} of {total_steps}; "
                         f"raise --epochs/--steps to train further")

    print(f"steps      : {total_steps} ({args.epochs} epochs x {args.steps_per_epoch})"
          if not args.steps else f"steps      : {total_steps}")
    print(f"batch      : {args.batch_size} x {args.accum_steps} accum "
          f"= {args.batch_size * args.accum_steps} effective")
    print(f"schedule   : AdamW wd={args.weight_decay}, warmup {args.warmup_steps} "
          f"-> lr {args.lr:g}, then x{args.lr_decay} per epoch, clip {args.clip_grad}")
    print(f"loss       : thresholded SNR"
          + (f" + {args.lambda_sparse} * sparsity" if args.lambda_sparse else " (no sparsity, paper-faithful)"))
    print(f"dataset    : {len(ds)} items, {len(dl)} micro-batches per pass", flush=True)

    model.train()
    opt.zero_grad(set_to_none=True)
    step = start_step     # optimiser updates (the paper's unit)
    micro = 0             # micro-batches since the last update
    acc = {"loss": 0.0, "mixit": 0.0, "sparse": 0.0, "gnorm": 0.0, "n": 0}
    t0 = time.time()
    done = False

    while not done:
        for mom, mixtures in dl:
            # mom [B,C,L] = sum of two mixtures; mixtures [B,2,C,L] = the two targets
            mom, mixtures = mom.to(device), mixtures.to(device)
            est = model(mom)                                  # [B,N,C,L]
            l_mixit, _ = efficient_mixit_loss(est, mixtures)  # Eq.3 in [1]
            l_sparse = sparsity_loss(est)
            loss = l_mixit + args.lambda_sparse * l_sparse
            # Divide by accum_steps so the accumulated gradient = average over the
            # effective batch (not the sum).
            (loss / args.accum_steps).backward()

            acc["loss"] += loss.item(); acc["mixit"] += l_mixit.item()
            acc["sparse"] += l_sparse.item(); acc["n"] += 1
            micro += 1
            if micro % args.accum_steps:
                continue                                      # keep accumulating

            if args.clip_grad > 0:
                acc["gnorm"] += float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                n = max(acc["n"], 1)
                per_step = (time.time() - t0) / max(step - start_step, 1)
                print(f"step {step:6d}/{total_steps} (epoch {step // args.steps_per_epoch:3d}) "
                      f"| loss {acc['loss'] / n:.4f} | mixit {acc['mixit'] / n:.4f} "
                      f"| sparse {acc['sparse'] / n:.4f} "
                      f"| gnorm {acc['gnorm'] / max(args.log_every, 1):.2f} "
                      f"| lr {sched.get_last_lr()[0]:.2e} "
                      f"| {per_step:.2f}s/step eta {(total_steps - step) * per_step / 3600:.1f}h",
                      flush=True)
                acc = {k: 0.0 for k in acc}

            if step % save_every == 0:
                save_all(model, opt, sched, step, args)

            if step >= total_steps:
                done = True
                break

    save_all(model, opt, sched, step, args)
    print(f"saved: {args.out} (weights), {args.ckpt_out} (resumable state)")


if __name__ == "__main__":
    main()

# References:
# [1] K. Saijo and Y. Bando, "Is MixIT Really Unsuitable for Correlated Sources?
#     Exploring MixIT for Unsupervised Pre-training in Music Source Separation," WASPAA 2025.
# [2] K. Saijo et al., "Task-Aware Unified Source Separation," ICASSP 2025.
#     https://github.com/merlresearch/tf-locoformer
