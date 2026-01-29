# torch.rand(B, C, T, dtype=torch.float32)  # Input shape: (1, 1, 48000 * 12)
import torch
import torch.nn as nn
import torchaudio

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=48000,
                n_fft=1536,
                hop_length=768,
                f_min=20,
                f_max=20000
            )
        )

    def forward(self, x):
        return self.transform(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((1, 1, 48000 * 12))

