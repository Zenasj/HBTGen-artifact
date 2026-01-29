# torch.rand(1, 1, 3000, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torchaudio

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, window_fn=torch.hann_window).to('cuda')

    def forward(self, x):
        return self.spectrogram(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 1, 3000).to('cuda')

