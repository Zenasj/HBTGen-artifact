# torch.rand(B, T, dtype=torch.float32)  # Input shape is [batch_size, signal_length]
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_window', torch.hann_window(window_length=320))  # Ensure window is part of model state

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        x = signals.stft(
            n_fft=512,
            hop_length=160,
            win_length=320,
            return_complex=True,
            window=self._window,
            pad_mode="constant",  # Matches original implementation despite ONNX export challenges
        )
        return x

def my_model_function():
    # Returns initialized MyModel instance
    return MyModel()

def GetInput():
    # Returns random input tensor matching [B, T] shape
    return torch.randn(2, 16000)  # Batch size 2, 16000-sample audio signals

