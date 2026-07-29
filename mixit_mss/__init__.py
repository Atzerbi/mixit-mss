"""mixit_mss: MixIT pre-training layer for music source separation.

Provides the MixIT layer (loss, mixture-of-mixtures dataset, channel selection,
two-stage training) on top of MERL's real BS-Locoformer, vendored under
mixit_mss/bslocoformer/ (Apache-2.0). See README.md.
"""

__version__ = "0.2.0"

from .losses import efficient_mixit_loss, sparsity_loss, negative_snr, exhaustive_mixit_loss
from .separator_adapter import (SeparatorAdapter, build_stub_adapter,
                                build_bslocoformer_adapter)
from .stft import STFT
from .datasets import MoMDataset, MUSDBDataset
from .fma import MoMFMADataset, build_manifest, scan_track_for_segments
from .channel_selection import select_channels
from .sad import sad_scan_track, build_musdb_sad_manifest
