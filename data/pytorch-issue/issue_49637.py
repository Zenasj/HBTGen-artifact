import torch

# spectrum represents complex data with trailing dimension, as for old torch.irfft
# ie, spectrum.size() --> (N, C, D, H, W, 2) for my case

# Old
signal = torch.irfft(spectrum, 3, normalized=True, signal_sizes=(d, h, w))

# Silent bug
signal = torch.fft.irfftn(spectrum, s=(d, h, w), dim=(2, 3, 4), norm='ortho')

# Fixed
spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
signal = torch.fft.irfftn(spectrum, s=(d, h, w), dim=(2, 3, 4), norm='ortho')