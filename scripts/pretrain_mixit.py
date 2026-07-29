"""Stage 1 — MixIT pre-training on unlabeled data (FMA).

Usage (stub model, for verification):
    python scripts/pretrain_mixit.py --stub --synthetic --steps 5

Real usage:
    python scripts/pretrain_mixit.py --clip_dir /path/to/fma --n_srcs 12
"""

import argparse
import torch
from torch.utils.data import DataLoader

from mixit_mss.datasets import MoMDataset
from mixit_mss.fma import MoMFMADataset
from mixit_mss.losses import efficient_mixit_loss, sparsity_loss
from mixit_mss.separator_adapter import build_stub_adapter, build_bslocoformer_adapter


def build_model(args, device):
    """Builds the separator.

    Default: REAL BS-Locoformer (vendored from MERL in mixit_mss/bslocoformer).
    --stub : uses the TF-domain stub to validate the pipeline without the large model.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default=None, help="raw folder (legacy MoMDataset)")
    ap.add_argument("--manifest", default=None, help="FMA manifest JSON (paper recipe)")
    ap.add_argument("--train_seconds", type=float, default=6.0,
                    help="training input length (paper: 6 s)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--use_fma", action="store_true", help="use FMA loader path")
    ap.add_argument("--n_srcs", type=int, default=12)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--stereo", action="store_true", default=True)
    ap.add_argument("--mono", dest="stereo", action="store_false")
    ap.add_argument("--stub", action="store_true", help="use the stub instead of the real model")
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--stft_size", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--segment_len", type=int, default=32000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_sparse", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="pretrained_mixit.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.manifest is not None or (args.synthetic and args.use_fma):
        ds = MoMFMADataset(manifest_path=args.manifest,
                           train_input_seconds=args.train_seconds,
                           _synthetic=args.synthetic)
    else:
        ds = MoMDataset(clip_dir=args.clip_dir, segment_len=args.segment_len,
                        n_channels=args.channels, _synthetic=args.synthetic)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = build_model(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    while step < args.steps:
        for mom, mixtures in dl:
            mom, mixtures = mom.to(device), mixtures.to(device)
            est = model(mom)                                  # [B,N,C,L]
            l_mixit, _ = efficient_mixit_loss(est, mixtures)
            l_sparse = sparsity_loss(est)
            loss = l_mixit + args.lambda_sparse * l_sparse
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 10 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f} "
                      f"| mixit {l_mixit.item():.4f} | sparse {l_sparse.item():.4f}")
            step += 1
            if step >= args.steps:
                break

    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
