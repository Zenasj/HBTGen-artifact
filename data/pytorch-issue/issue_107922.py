# torch.rand(1, 1, 576000, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn
import torchaudio

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1536, hop_length=768, f_min=20, f_max=20000)
        )

    def forward(self, x1):
        x1 = self.transform(x1)
        return x1

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((1, 1, 12 * 48000), dtype=torch.float32)

