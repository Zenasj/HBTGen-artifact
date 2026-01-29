# torch.rand(4, 4, 257, 594, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, speechs_hat):
        n_frames = speechs_hat.shape[-1]
        return torch.einsum(
            "nift, njft -> nfij",
            speechs_hat,
            speechs_hat.conj() / n_frames
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, 257, 594, dtype=torch.complex64)

