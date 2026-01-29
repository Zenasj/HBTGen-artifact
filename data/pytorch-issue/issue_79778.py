# torch.rand(B, L, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        stft = torch.stft(
            x,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            center=False,
            return_complex=True
        )
        output = torch.istft(
            stft,
            n_fft=1024,
            hop_length=256,
            center=False
        )
        input_length = x.shape[-1]
        output_length = output.shape[-1]
        return torch.tensor(output_length == input_length, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 25600)

