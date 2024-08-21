import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvergenceLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, magnitude_pred, magnitude_true):
        return torch.norm(magnitude_true - magnitude_pred, p="fro") / torch.norm(magnitude_true, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, magnitude_pred, magnitude_true):
        return F.l1_loss(torch.log(magnitude_true), torch.log(magnitude_pred))


class STFTLoss(torch.nn.Module):

    def __init__(self, n_fft, hop_length, win_length, window):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

        self.register_buffer("window", getattr(torch, window)(win_length))

    def _magnitude(self, x):
        x_stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=self.window.to(x.device),
            return_complex=True
        )
        magnitude_squared = torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-7)
        magnitude = torch.sqrt(magnitude_squared)

        return magnitude.transpose(2, 1)

    def forward(self, y_pred, y_true):
        magnitude_pred = self._magnitude(y_pred)
        magnitude_true = self._magnitude(y_true)

        sc_loss = self.spectral_convergence_loss(magnitude_pred, magnitude_true)
        mag_loss = self.log_stft_magnitude_loss(magnitude_pred, magnitude_true)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):

    def __init__(
            self, n_ffts, hop_lengths, win_lengths,
            window, sc_lambda=0.1, mag_lambda=0.1
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda

        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, y_pred, y_true):
        if len(y_pred.shape) == 3:
            y_pred = y_pred.view(-1, y_pred.size(2))
            y_true = y_true.view(-1, y_true.size(2))
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(y_pred, y_true)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss *= self.sc_lambda
        sc_loss /= len(self.stft_losses)
        mag_loss *= self.mag_lambda
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


with open('config/config.json') as f:
    data = f.read()
config = json.loads(data)['train']['loss']

reconstruction_loss = None
if config['reconstruction'].lower() == 'mse_loss':
    reconstruction_loss = nn.MSELoss()
elif config['reconstruction'].lower() == 'mae_loss':
    reconstruction_loss = nn.L1Loss()

mrstft_loss = MultiResolutionSTFTLoss(
    config['stft']['n_ffts'],
    config['stft']['hop_lengths'],
    config['stft']['win_lengths'],
    config['stft']['window']
)
