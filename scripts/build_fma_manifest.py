"""Offline pre-processing: build the FMA segment manifest.

Scans an FMA audio folder, applies the sample-rate policy (drop < 44.1 kHz,
downsample > 44.1 kHz) and the silence rule (10 s segments, 5 s overlap, discard
if > 5 s of silence measured over 1 s intervals), and writes a JSON manifest.

Usage:
    python scripts/build_fma_manifest.py --clip_dir /path/to/fma_large \
        --out fma_manifest.json --silence_threshold 1e-4

Then pre-train with:
    python scripts/pretrain_mixit.py --manifest fma_manifest.json --n_srcs 12
"""

import argparse
from mixit_mss.fma import build_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True, help="FMA audio root")
    ap.add_argument("--out", default="fma_manifest.json")
    ap.add_argument("--silence_threshold", type=float, default=1e-4,
                    help="mean power below this = silent interval (tune to your data)")
    args = ap.parse_args()
    build_manifest(args.clip_dir, args.out, silence_threshold=args.silence_threshold)

if __name__ == "__main__":
    main()
