# torch.rand(1, 1, 12 * 48000, dtype=torch.float32)
import torch
from torch import nn
import torchaudio

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=48000,
                n_fft=1536,
                hop_length=768,
                f_min=20,
                f_max=20000
            )
        )

    def forward(self, x1):
        return self.transform(x1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 12 * 48000, dtype=torch.float32)

