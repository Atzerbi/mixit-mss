"""Adapter between the real BS-Locoformer and the pipeline contract.

Clean contract required by the rest of the package:
    forward(mom) : mom [B, C, L]  ->  sources [B, N, C, L]   (waveforms)

REAL signature of the standalone BS-Locoformer (verified from MERL source):
    input  : complex spectrum  [B, T, F] (mono)  or  [B, C, T, F] (stereo)
    output : complex spectrum  [B, N, T, F]      or  [B, N, C, T, F] (stereo)
    (no internal STFT: it must be done outside, see stft.py)

This adapter wraps STFT -> model -> iSTFT and produces waveforms, so the MixIT
loss (which compares waveforms) stays unchanged.
"""

import torch
import torch.nn as nn

from .stft import STFT


class SeparatorAdapter(nn.Module):
    def __init__(self, separator, n_srcs, n_channels=2, stereo=True,
                 n_fft=2048, hop_length=512):
        """
        separator  : BSLocoformerSeparator instance (or TF-domain compatible).
        n_srcs     : N (model's num_spk).
        n_channels : C (2 for stereo).
        stereo     : must match the BS-Locoformer `stereo` flag.
                     True  -> model receives [B, C, T, F], emits [B, N, C, T, F]
                     False -> mono model; the C channels are processed as a batch.
        """
        super().__init__()
        self.model = separator
        self.n_srcs = n_srcs
        self.C = n_channels
        self.stereo = stereo
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

    def forward(self, mom):
        # mom: [B, C, L]
        B, C, L = mom.shape
        spec = self.stft.encode(mom)          # [B, C, T, F]

        if self.stereo:
            out_spec = self.model(spec)        # [B, N, C, T, F]
            N = out_spec.shape[1]
            flat = out_spec.reshape(B * N, C, out_spec.shape[-2], out_spec.shape[-1])
            wav = self.stft.decode(flat, length=L)          # [B*N, C, L]
            return wav.reshape(B, N, C, L).contiguous()
        else:
            spec_mono = spec.reshape(B * C, spec.shape[-2], spec.shape[-1])  # [B*C, T, F]
            out_spec = self.model(spec_mono)                # [B*C, N, T, F]
            N = out_spec.shape[1]
            flat = out_spec.reshape(B * C * N, 1, out_spec.shape[-2], out_spec.shape[-1])
            wav = self.stft.decode(flat, length=L)          # [B*C*N, 1, L]
            wav = wav.reshape(B, C, N, L).permute(0, 2, 1, 3)   # [B, N, C, L]
            return wav.contiguous()


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def build_bslocoformer_adapter(n_srcs=12, n_channels=2, stereo=True,
                               n_layers=6, emb_dim=128, sample_rate=44100,
                               stft_size=2048, hop_length=512, **kw):
    """Builds a SeparatorAdapter with the REAL BS-Locoformer.

    n_srcs -> model's num_spk (12 for MixIT pre-training).
    NOTE: the model's stft_size and the external STFT n_fft must match.
    """
    from .bslocoformer import BSLocoformerSeparator
    model = BSLocoformerSeparator(
        num_spk=n_srcs, n_layers=n_layers, emb_dim=emb_dim,
        norm_type=kw.pop("norm_type", "rmsgroupnorm"),
        num_groups=kw.pop("num_groups", 8),
        n_heads=kw.pop("n_heads", 8),
        attention_dim=kw.pop("attention_dim", 128),
        pos_enc=kw.pop("pos_enc", "rope"),
        ffn_type=kw.pop("ffn_type", "swiglu_conv1d"),
        ffn_hidden_dim=kw.pop("ffn_hidden_dim", 384),
        conv1d_kernel=kw.pop("conv1d_kernel", 8),
        conv1d_shift=kw.pop("conv1d_shift", 1),
        sample_rate=sample_rate, stft_size=stft_size,
        masking=kw.pop("masking", True), stereo=stereo,
    )
    return SeparatorAdapter(model, n_srcs=n_srcs, n_channels=n_channels,
                            stereo=stereo, n_fft=stft_size, hop_length=hop_length)

# ---------------------------------------------------------------------------
# TF-domain stub for fast tests without building the large model
# ---------------------------------------------------------------------------
class _StubTFSeparator(nn.Module):
    """Mimics the BS-Locoformer TF-domain signature: complex spectrum in/out."""
    def __init__(self, n_srcs, stereo=True):
        super().__init__()
        self.n_srcs = n_srcs
        self.stereo = stereo
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, spec):
        if self.stereo:
            B, C, T, Fr = spec.shape
            out = spec.unsqueeze(1).expand(B, self.n_srcs, C, T, Fr) * self.scale
        else:
            B, T, Fr = spec.shape
            out = spec.unsqueeze(1).expand(B, self.n_srcs, T, Fr) * self.scale
        return out.contiguous()


def build_stub_adapter(n_srcs=12, n_channels=2, stereo=True,
                       n_fft=2048, hop_length=512):
    """SeparatorAdapter with a TF-domain stub, for fast end-to-end tests."""
    stub = _StubTFSeparator(n_srcs=n_srcs, stereo=stereo)
    return SeparatorAdapter(stub, n_srcs=n_srcs, n_channels=n_channels,
                            stereo=stereo, n_fft=n_fft, hop_length=hop_length)
