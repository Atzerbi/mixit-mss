"""Offline SAD pass over MUSDB training stems (BSRNN recipe, Sec. IV-A1).

Applies unsupervised energy-based source activity detection to each stem of each
MUSDB track, recording the salient (non-silent) segment offsets. Saijo & Bando
apply this to the MUSDB training data before dynamic mixing.

Usage:
    python scripts/build_musdb_sad.py --musdb_root /path/to/musdb18hq/train \
        --out musdb_sad.json --segment_seconds 6

Then fine-tune with:
    python scripts/finetune_musdb.py --musdb_root ... --sad_manifest musdb_sad.json ...
"""

import argparse
from mixit_mss.sad import build_musdb_sad_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb_root", required=True, help="MUSDB train root (per-track folders)")
    ap.add_argument("--out", default="musdb_sad.json")
    ap.add_argument("--segment_seconds", type=float, default=6.0)
    args = ap.parse_args()
    build_musdb_sad_manifest(args.musdb_root, args.out,
                             segment_seconds=args.segment_seconds)


if __name__ == "__main__":
    main()
