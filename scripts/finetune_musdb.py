"""Stage 2 - Channel selection + supervised fine-tuning on MUSDB18.

Flow:
    1. load the MixIT pre-trained model (N=12 unlabeled outputs);
    2. channel selection: map the 12 outputs -> 4 VDBO stems on a validation set
       (done ONCE; the resulting map is stored in the checkpoint and reused on resume);
    3. supervised fine-tuning with direct loss (or PIT) on the 4 selected channels.

Usage (paper recipe):
    python scripts/finetune_musdb.py --musdb_root .../train --sad_manifest musdb_sad.json \
        --pretrained pretrained_mixit.pth --batch_size 1 --accum_steps 8

Resume an interrupted run (see pretrain_mixit.py for the same mechanism). The
channel map is restored from the checkpoint, so channel selection is NOT re-run:
    python scripts/finetune_musdb.py --musdb_root .../train --sad_manifest musdb_sad.json \
        --batch_size 1 --accum_steps 8 --resume
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from mixit_mss.datasets import MUSDBDataset, STEMS
from mixit_mss.separator_adapter import build_stub_adapter, build_bslocoformer_adapter
from mixit_mss.channel_selection import select_channels
from mixit_mss.pit import direct_loss, pit_loss


def build_model(args, device):
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


def atomic_save(obj, path):
    """Write to a temp file then rename, so a job killed mid-write never leaves a
    truncated checkpoint (the previous one stays intact until the new one is done)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_all(model, opt, mapping, step, args):
    """Weights + channel map (for inference/eval) + full training state (for --resume)."""
    # deliverable: weights + channel map, consumed by inference.py / evaluation.py
    atomic_save({"state_dict": model.state_dict(), "channel_map": mapping}, args.out)
    # resume state: everything needed to continue exactly where it stopped
    atomic_save({"state_dict": model.state_dict(),
                 "optimizer": opt.state_dict(),
                 "channel_map": mapping,
                 "step": step,
                 "args": vars(args)}, args.ckpt_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb_root", default=None)
    ap.add_argument("--sad_manifest", default=None, help="MUSDB SAD manifest JSON")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pretrained", default=None, help="MixIT pre-training weights to start from")
    ap.add_argument("--n_srcs", type=int, default=12)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--stereo", action="store_true", default=True)
    ap.add_argument("--mono", dest="stereo", action="store_false")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--stft_size", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--segment_len", type=int, default=32000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="micro-batches per optimiser step; effective batch = batch_size*accum_steps")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip_grad", type=float, default=5.0, help="max gradient L2-norm (0 = off)")
    ap.add_argument("--steps", type=int, default=100, help="total optimiser steps")
    ap.add_argument("--use_pit", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0, help="checkpoint every N steps; 0 = every 100")
    ap.add_argument("--out", default="finetuned_musdb.pth")
    ap.add_argument("--ckpt_out", default=None, help="full training-state file (default: <out>.ckpt)")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="continue training; bare --resume picks up <out>.ckpt")
    ap.add_argument("--start_step", type=int, default=None,
                    help="only for a weights-only --resume: where the interrupted run stopped")
    args = ap.parse_args()
    if args.ckpt_out is None:
        args.ckpt_out = args.out + ".ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ resume --
    start_step = 0
    mapping = None
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = args.ckpt_out if os.path.exists(args.ckpt_out) else args.out
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise SystemExit(f"--resume: no such checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])       # strict: catches hyperparam mismatch
            mapping = ckpt.get("channel_map")               # reuse the map -> skip re-selection
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
                start_step = ckpt.get("step", 0)
                print(f"resume     : {resume_path} -> step {start_step} (full state)")
            else:
                if args.start_step is None:
                    raise SystemExit("--resume from a weights-only checkpoint needs --start_step")
                start_step = args.start_step
                print(f"resume     : {resume_path} -> step {start_step} (weights only; "
                      "optimiser state reset)")
        else:
            raise SystemExit("--resume: unrecognized checkpoint format")
        if mapping is None:
            print("WARNING: resumed checkpoint has no channel_map; will run channel selection")
    elif args.pretrained:
        sd = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)             # 12-output pre-training weights
        print(f"loaded pre-trained weights from {args.pretrained}")

    if start_step >= args.steps:
        raise SystemExit(f"nothing to do: resumed at step {start_step} of {args.steps}; "
                         "raise --steps to continue")

    # ------------------------------------------------------------------ data ----
    train_ds = MUSDBDataset(root=args.musdb_root, segment_len=args.segment_len,
                            n_channels=args.channels, dynamic_mixing=True,
                            sad_manifest=args.sad_manifest, _synthetic=args.synthetic)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # -------------------------------------------------------- channel selection -
    # Run it only if we don't already have a map from a resumed checkpoint. This
    # avoids both wasted compute and a possibly different map on restart.
    if mapping is None:
        val_ds = MUSDBDataset(root=args.musdb_root, segment_len=args.segment_len,
                              n_channels=args.channels, dynamic_mixing=False,
                              _synthetic=args.synthetic)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True)
        print("running channel selection...")
        mapping, _ = select_channels(model, val_dl, n_srcs=args.n_srcs,
                                     device=device, max_batches=4)
        print("stem->channel map:", mapping)
    else:
        print("stem->channel map (from checkpoint):", mapping)
    sel = [mapping[s] for s in STEMS]

    save_every = args.save_every or 100
    print(f"batch      : {args.batch_size} x {args.accum_steps} accum "
          f"= {args.batch_size * args.accum_steps} effective")
    print(f"steps      : {start_step} -> {args.steps} | loss "
          f"{'PIT' if args.use_pit else 'direct'} | clip {args.clip_grad}", flush=True)

    # ------------------------------------------------------------- training -----
    model.train()
    opt.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    acc = {"loss": 0.0, "gnorm": 0.0, "n": 0}
    t0 = time.time()
    done = False

    while not done:
        for mix, targets in train_dl:
            mix, targets = mix.to(device), targets.to(device)
            est = model(mix)                       # [B, N, C, L]
            est_sel = est[:, sel]                  # [B, 4, C, L]
            if args.use_pit:
                loss, _ = pit_loss(est_sel, targets)
            else:
                loss = direct_loss(est_sel, targets)
            (loss / args.accum_steps).backward()
            acc["loss"] += loss.item(); acc["n"] += 1
            micro += 1
            if micro % args.accum_steps:
                continue

            if args.clip_grad > 0:
                acc["gnorm"] += float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad))
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                n = max(acc["n"], 1)
                per = (time.time() - t0) / max(step - start_step, 1)
                print(f"step {step:6d}/{args.steps} | loss {acc['loss']/n:.4f} "
                      f"| gnorm {acc['gnorm']/max(args.log_every,1):.2f} "
                      f"| {per:.2f}s/step eta {(args.steps-step)*per/3600:.1f}h", flush=True)
                acc = {k: 0.0 for k in acc}

            if step % save_every == 0:
                save_all(model, opt, mapping, step, args)

            if step >= args.steps:
                done = True
                break

    save_all(model, opt, mapping, step, args)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
