"""Test inference: chunked overlap-add reconstruction + both scenarios."""

import os
import tempfile
import torch
from mixit_mss.inference import (Separator, separate_waveform, separate,
                                 save_stems, STEMS)
from mixit_mss.separator_adapter import build_stub_adapter


def _stub_sep(n_srcs=12, sr=8000, channel_map=None):
    model = build_stub_adapter(n_srcs=n_srcs, n_channels=2, stereo=True,
                               n_fft=512, hop_length=128)
    model.eval()
    return Separator(model, n_srcs=n_srcs, sr=sr, channel_map=channel_map, device="cpu")


def test_overlap_add_shapes():
    sep = _stub_sep()
    wav = torch.randn(2, 3 * sep.sr)          # ~3 s stereo
    out = separate_waveform(sep, wav, chunk_seconds=1.0, overlap=0.5)
    assert out.shape == (sep.n_srcs, 2, wav.shape[1]), out.shape
    print(f"[ok] overlap-add: in {tuple(wav.shape)} -> out {tuple(out.shape)}")


def test_overlap_add_reconstruction():
    """The stub echoes input as every source. Overlap-add of an echo should
    reconstruct the input closely (checks the cross-fade normalization)."""
    sep = _stub_sep()
    wav = torch.randn(2, 2 * sep.sr)
    out = separate_waveform(sep, wav, chunk_seconds=1.0, overlap=0.5)
    # stub multiplies spectrum by a (learnable) scale ~1; channel 0 ~ input
    err = (out[0] - wav).abs().mean().item()
    ref = wav.abs().mean().item()
    print(f"[info] recon mean-abs-err={err:.4f} vs signal level {ref:.4f}")
    assert err < 0.5 * ref, "overlap-add reconstruction too far from input"
    print("[ok] overlap-add reconstruction")


def test_raw_scenario():
    sep = _stub_sep(n_srcs=12)
    with tempfile.TemporaryDirectory() as d:
        import torchaudio
        p = os.path.join(d, "song.wav")
        torchaudio.save(p, torch.randn(2, sep.sr), sep.sr)
        stems = separate(sep, p, chunk_seconds=1.0)
        assert len(stems) == 12 and "src00" in stems
        paths = save_stems(stems, os.path.join(d, "out"), sr=sep.sr)
        assert all(os.path.exists(x) for x in paths.values())
    print(f"[ok] RAW scenario: {len(stems)} channels written")


def test_finetuned_scenario():
    cmap = {"vocals": 0, "drums": 3, "bass": 7, "other": 11}
    sep = _stub_sep(n_srcs=12, channel_map=cmap)
    assert sep.is_finetuned
    with tempfile.TemporaryDirectory() as d:
        import torchaudio
        p = os.path.join(d, "song.wav")
        torchaudio.save(p, torch.randn(2, sep.sr), sep.sr)
        stems = separate(sep, p, chunk_seconds=1.0)
        assert set(stems.keys()) == set(STEMS)
    print(f"[ok] FINE-TUNED scenario: stems = {list(stems.keys())}")


if __name__ == "__main__":
    test_overlap_add_shapes()
    test_overlap_add_reconstruction()
    test_raw_scenario()
    test_finetuned_scenario()
    print("\nAll inference tests passed.")
