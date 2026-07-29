"""Stage 2 — Channel selection + supervised fine-tuning on MUSDB18.

Usage (stub, verification):
    python scripts/finetune_musdb.py --stub --synthetic --steps 5

Flow:
    1. load the MixIT pre-trained model (N outputs);
    2. channel selection: map N -> 4 VDBO stems on a validation set;
    3. supervised fine-tuning with direct loss (or PIT) on the 4 selected channels.
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb_root", default=None)
    ap.add_argument("--sad_manifest", default=None, help="MUSDB SAD manifest JSON")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pretrained", default=None)
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
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--use_pit", action="store_true")
    ap.add_argument("--out", default="finetuned_musdb.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(args, device)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"loaded pre-trained weights from {args.pretrained}")

    train_ds = MUSDBDataset(root=args.musdb_root, segment_len=args.segment_len,
                            n_channels=args.channels, dynamic_mixing=True,
                            sad_manifest=args.sad_manifest,
                            _synthetic=args.synthetic)
    val_ds = MUSDBDataset(root=args.musdb_root, segment_len=args.segment_len,
                          n_channels=args.channels, dynamic_mixing=False,
                          _synthetic=args.synthetic)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True)

    # --- Channel selection ---
    print("running channel selection...")
    mapping, counts = select_channels(model, val_dl, n_srcs=args.n_srcs,
                                      device=device, max_batches=4)
    print("stem->channel map:", mapping)
    sel = [mapping[s] for s in STEMS]     # indices of the 4 selected channels

    # --- Supervised fine-tuning ---
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    step = 0
    while step < args.steps:
        for mix, targets in train_dl:
            mix, targets = mix.to(device), targets.to(device)
            est = model(mix)                       # [B, N, C, L]
            est_sel = est[:, sel]                  # [B, 4, C, L] selected channels
            if args.use_pit:
                loss, _ = pit_loss(est_sel, targets)
            else:
                loss = direct_loss(est_sel, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 10 == 0:
                print(f"step {step:5d} | loss {loss.item():.4f}")
            step += 1
            if step >= args.steps:
                break

    torch.save({"state_dict": model.state_dict(), "channel_map": mapping}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
