"""STFT encoder/decoder external to the separator.

The standalone BS-Locoformer works ENTIRELY in the TF domain: it takes a complex
spectrogram and returns complex spectrograms. The (i)STFT must therefore be handled
here, outside the model. This module bridges between waveforms (which the MixIT loss
compares) and spectrograms (which the model processes).

Conventions:
  stereo waveform : [B, C, L]
  stereo spectrum : [B, C, T, F]  (complex)   <- input expected by BS-Locoformer
"""

import torch


class STFT:
    def __init__(self, n_fft=2048, hop_length=512, window="hann", center=True):
        self.n_fft = n_fft
        self.hop = hop_length
        self.center = center
        if window == "hann":
            self.window = torch.hann_window(n_fft)
        else:
            raise ValueError(window)

    def _win(self, device):
        if self.window.device != device:
            self.window = self.window.to(device)
        return self.window

    def encode(self, wav):
        """[B, C, L] -> complex spectrum [B, C, T, F]."""
        B, C, L = wav.shape
        x = wav.reshape(B * C, L)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          window=self._win(wav.device), center=self.center,
                          return_complex=True)                  # [B*C, F, T]
        spec = spec.transpose(1, 2)                             # [B*C, T, F]
        F_, T_ = spec.shape[-1], spec.shape[-2]
        return spec.reshape(B, C, T_, F_)                       # [B, C, T, F]

    def decode(self, spec, length=None):
        """complex spectrum [B, C, T, F] -> waveform [B, C, L]."""
        B, C, T_, F_ = spec.shape
        x = spec.reshape(B * C, T_, F_).transpose(1, 2)         # [B*C, F, T]
        wav = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop,
                          window=self._win(spec.device), center=self.center,
                          length=length)                        # [B*C, L]
        return wav.reshape(B, C, -1)                            # [B, C, L]
