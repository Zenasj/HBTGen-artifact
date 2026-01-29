# torch.rand(B, C, 1, 1, dtype=torch.float32)  # e.g., B=5, C=6 for this case
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Convert 4D input to 2D (B, C) for multinomial
        x = x.view(x.size(0), -1)
        return torch.multinomial(x, 2, replacement=False)

def my_model_function():
    return MyModel()

def GetInput():
    # Create a 5x6 input with only the 4th column (index 3) set to 1
    weights = torch.zeros(5, 6, 1, 1)
    weights[:, 3, :, :] = 1.0
    return weights.cuda()  # Matches the CUDA context in the issue's example

