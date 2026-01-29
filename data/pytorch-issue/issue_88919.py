# torch.rand(B, T, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=None, window='hann'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window_type = window  # Stores the window name as a string

    def forward(self, input):
        # Handle window creation based on string input
        if isinstance(self.window_type, str):
            windows = {
                "hann": torch.hann_window,
                "hamming": torch.hamming_window,
                "blackman": torch.blackman_window
            }
            if self.window_type not in windows:
                raise ValueError(f"Invalid window '{self.window_type}'. Available: {list(windows.keys())}")
            window_func = windows[self.window_type]
            window = window_func(self.win_length, device=input.device, dtype=input.dtype)
        else:
            window = self.window_type  # Assume Tensor type

        # Apply STFT with inferred window and parameters
        return torch.stft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )

def my_model_function():
    # Returns MyModel with default parameters (Hann window by default)
    return MyModel()

def GetInput():
    # Returns random tensor matching STFT input requirements (batch, time)
    return torch.randn(2, 4096, dtype=torch.float32)

