# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        s4 = x.size(0)
        s5 = x.size(1)
        part1 = s4 // 21
        part2 = s4 // part1  # (s4 // (s4//21))
        term1 = part1 * part2
        part3 = s5 // 120
        part4 = s5 // part3  # (s5 // (s5//120))
        term2 = part1 * part4
        total = (term1 + term2) // 141
        # Create a tensor using total to trigger symbolic shape computation
        return x.view(-1, total).sum()  # Dummy output for gradient computation

def my_model_function():
    return MyModel()

def GetInput():
    # Dimensions divisible by 21, 120, and 141 to avoid division remainder issues
    B = 21
    C = 120
    H = 141
    W = 3
    return torch.rand(B, C, H, W, dtype=torch.float32)

