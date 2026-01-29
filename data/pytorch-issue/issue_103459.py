# torch.rand(5, 8192, 96, 96, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12  # As per the workaround's epsilon addition

    def forward(self, x):
        # Compute original (buggy) gumbel_softmax
        gumbels_buggy = - torch.empty_like(x).exponential_().log()
        gumbels_buggy = (x + gumbels_buggy) / 1.0  # Tau=1.0 by default
        y_buggy = gumbels_buggy.softmax(dim=-1)

        # Compute fixed version (with epsilon)
        gumbels_fixed = torch.empty_like(x).exponential_()
        gumbels_fixed += self.eps
        gumbels_fixed = -gumbels_fixed.log()
        gumbels_fixed = (x + gumbels_fixed) / 1.0
        y_fixed = gumbels_fixed.softmax(dim=-1)

        # Compare outputs for closeness (NaNs in y_buggy will cause failure)
        return torch.isclose(y_buggy, y_fixed, atol=1e-5).all()

def my_model_function():
    return MyModel()

def GetInput():
    torch.manual_seed(1234)
    x = torch.rand(5, 8192, 96, 96, dtype=torch.float32) * 2 - 1
    x *= 255
    return x

