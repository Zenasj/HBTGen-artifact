# torch.rand(B, N, dtype=torch.float16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_samples=1, replacement=True):
        super().__init__()
        self.num_samples = num_samples
        self.replacement = replacement

    def forward(self, input):
        return torch.multinomial(input, self.num_samples, self.replacement)

def my_model_function():
    return MyModel()

def GetInput():
    # Using test case dimensions from the issue comments
    B, N = 2, 3
    return torch.rand(B, N, dtype=torch.float16, device='cuda')

