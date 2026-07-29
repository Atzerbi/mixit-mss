"""Pipeline tests: shapes, backward, solver validation, and real BS-Locoformer.

Run: python tests/test_pipeline.py
"""

import torch
from mixit_mss.losses import (efficient_mixit_loss, exhaustive_mixit_loss,
                              sparsity_loss, negative_snr)
from mixit_mss.separator_adapter import build_stub_adapter, build_bslocoformer_adapter
from mixit_mss.datasets import MoMDataset, MUSDBDataset
from mixit_mss.channel_selection import select_channels
from mixit_mss.pit import pit_loss, direct_loss


def test_shapes_and_backward():
    B, C, L, N = 2, 2, 16384, 12
    for stereo in [True, False]:
        model = build_stub_adapter(n_srcs=N, n_channels=C, stereo=stereo,
                                   n_fft=512, hop_length=128)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        mom = torch.randn(B, C, L)
        mixtures = torch.randn(B, 2, C, L)
        est = model(mom)
        assert est.shape == (B, N, C, L), f"stereo={stereo}: {est.shape}"
        loss, A = efficient_mixit_loss(est, mixtures)
        assert A.shape == (B, 2, N)
        opt.zero_grad(); (loss + 0.1 * sparsity_loss(est)).backward(); opt.step()
    print("[ok] shapes & backward (stereo True/False)")


def test_efficient_matches_exhaustive():
    torch.manual_seed(0)
    B, C, L, N, M = 8, 1, 2000, 4, 2
    true_src = torch.randn(B, N, C, L)
    assign = torch.randint(0, M, (B, N))
    mixtures = torch.zeros(B, M, C, L)
    for b in range(B):
        for n in range(N):
            mixtures[b, assign[b, n]] += true_src[b, n]
    l_eff, _ = efficient_mixit_loss(true_src, mixtures)
    l_exh, _ = exhaustive_mixit_loss(true_src, mixtures)
    gap = (l_eff - l_exh).item()
    print(f"[info] efficient={l_eff.item():.4f}  exhaustive={l_exh.item():.4f}  gap={gap:.4f}")
    assert gap >= -1e-3 and gap < 5.0
    print("[ok] efficient vs exhaustive")


def test_datasets():
    ds = MoMDataset(segment_len=4000, n_channels=2, _synthetic=True)
    mom, mix = ds[0]
    assert mom.shape == (2, 4000) and mix.shape == (2, 2, 4000)
    mds = MUSDBDataset(segment_len=4000, n_channels=2, _synthetic=True)
    m, t = mds[0]
    assert m.shape == (2, 4000) and t.shape == (4, 2, 4000)
    print("[ok] datasets")


def test_channel_selection_and_pit():
    B, C, L, N = 2, 2, 8192, 12
    model = build_stub_adapter(n_srcs=N, n_channels=C, stereo=True,
                               n_fft=512, hop_length=128)
    mds = MUSDBDataset(segment_len=L, n_channels=C, _synthetic=True)
    from torch.utils.data import DataLoader
    dl = DataLoader(mds, batch_size=B, drop_last=True)
    mapping, counts = select_channels(model, dl, n_srcs=N, max_batches=2)
    assert set(mapping.keys()) == {"vocals", "drums", "bass", "other"}
    assert len(set(mapping.values())) == 4
    est = torch.randn(B, 4, C, L)
    targets = torch.randn(B, 4, C, L)
    assert pit_loss(est, targets)[0].dim() == 0
    assert direct_loss(est, targets).dim() == 0
    print(f"[ok] channel selection {mapping} & pit")


def test_real_bslocoformer():
    """End-to-end pass with the REAL BS-Locoformer (small config)."""
    B, C, N = 1, 2, 4
    L = 16384
    model = build_bslocoformer_adapter(
        n_srcs=N, n_channels=C, stereo=True,
        n_layers=1, emb_dim=32, num_groups=1, n_heads=4, attention_dim=32,
        ffn_type=["swiglu_conv1d", "swiglu_conv1d"], ffn_hidden_dim=[32, 32],
        sample_rate=44100, stft_size=2048, hop_length=512,
    )
    mom = torch.randn(B, C, L)
    mixtures = torch.randn(B, 2, C, L)
    est = model(mom)
    assert est.shape == (B, N, C, L), f"real model est {est.shape}"
    loss, _ = efficient_mixit_loss(est, mixtures)
    loss = loss + 0.1 * sparsity_loss(est)
    loss.backward()
    print(f"[ok] BS-Locoformer REAL: est={tuple(est.shape)} loss={loss.item():.4f}")



def test_fma_segmentation_recipe():
    """Verifies the paper's segmentation: 10 s / 5 s overlap and the >5 s silence rule."""
    from mixit_mss.fma import scan_track_for_segments, MoMFMADataset
    sr = 44100
    # 30 s stereo track: first 20 s loud, last 10 s silent
    N = 30 * sr
    wav = torch.randn(2, N) * 0.5
    wav[:, 20 * sr:] = 0.0                      # 10 s of trailing silence
    offsets = scan_track_for_segments(wav, sr, silence_threshold=1e-4)
    # segments start at 0,5,10,15,20 s (start+10s <= 30s). A segment is kept iff
    # it has <=5 s silence. Segments starting at 0,5,10 s are fully loud (kept);
    # the one at 15 s spans 15-25 s = 5 s silence (kept, boundary); at 20 s spans
    # 20-30 s = 10 s silence (dropped).
    starts_s = sorted(o / sr for o in offsets)
    assert 0.0 in starts_s and 5.0 in starts_s and 10.0 in starts_s
    assert 20.0 not in starts_s, "fully-silent segment must be dropped"
    print(f"[ok] FMA segmentation: kept starts (s) = {starts_s}")

    # synthetic MoM loader path: 6 s training crop, stereo, MoM shapes
    ds = MoMFMADataset(_synthetic=True, train_input_seconds=6.0)
    mom, mixtures = ds[0]
    L6 = int(6.0 * sr)
    assert mom.shape == (2, L6) and mixtures.shape == (2, 2, L6)
    print(f"[ok] FMA MoM loader: mom={tuple(mom.shape)} mixtures={tuple(mixtures.shape)}")



def test_sad_recipe():
    """Verifies the BSRNN energy-based SAD: loud segments kept, silent ones rejected."""
    from mixit_mss.sad import sad_scan_track
    sr = 44100
    # 24 s stereo: first 12 s loud, last 12 s silent
    N = 24 * sr
    wav = torch.randn(2, N) * 0.5
    wav[:, 12 * sr:] = 0.0
    offsets = sad_scan_track(wav, sr, segment_seconds=6.0)
    starts_s = sorted(o / sr for o in offsets)
    # 6 s segments, 50% overlap -> starts at 0,3,6,9,12,15,18 s (start+6<=24).
    # Loud region is 0-12 s: segments fully inside (0,3,6) are salient; the 12 s
    # boundary onward is silent and must be rejected.
    assert 0.0 in starts_s and 3.0 in starts_s and 6.0 in starts_s
    assert 15.0 not in starts_s and 18.0 not in starts_s, "silent segments must be rejected"
    print(f"[ok] SAD: kept starts (s) = {starts_s}")

    # MUSDBDataset with a synthetic SAD manifest sampling path
    from mixit_mss.datasets import MUSDBDataset
    mds = MUSDBDataset(segment_len=8192, n_channels=2, dynamic_mixing=True,
                       _synthetic=True)
    mix, targets = mds[0]
    assert mix.shape == (2, 8192) and targets.shape == (4, 2, 8192)
    print("[ok] MUSDBDataset dynamic mixing (RMS-norm + gain + drop)")


if __name__ == "__main__":
    test_shapes_and_backward()
    test_efficient_matches_exhaustive()
    test_datasets()
    test_channel_selection_and_pit()
    test_real_bslocoformer()
    test_fma_segmentation_recipe()
    test_sad_recipe()
    print("\nAll tests passed.")
