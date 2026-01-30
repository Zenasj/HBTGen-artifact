import torch.nn as nn

import torch


class STFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._window = torch.hann_window(window_length=320)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,
            window=self._window,
            pad_mode="constant",
        )
        return x


m = STFTModel()

# Shape [B, T] audio signals
input_signals = torch.randn([2, 16000])

args = (input_signals,)
ep = torch.export.export(
    m,
    args,
)

# Successful
print(ep)

# Dynamic axis
# Fails
ep2 = torch.export.export(
    m,
    args,
    dynamic_shapes=[
        {
            0: torch.export.Dim("dim1"),
            1: torch.export.Dim("dim2"),
        },
    ],
)

print(ep2)