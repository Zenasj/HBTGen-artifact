# torch.rand(3, 2), torch.rand(2, 3), torch.rand(8000, dtype=torch.float).cuda() ← Inferred input shapes for mm and sort operations
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b, c = inputs
        # Mimic custom mm and sort operations from the issue
        mm_result = torch.mm(a, b)  # Matrix multiply (3x2 @ 2x3 → 3x3)
        sorted_c, _ = torch.sort(c)  # Sort 1D tensor of size 8000
        return mm_result, sorted_c

def my_model_function():
    return MyModel()

def GetInput():
    # Generate inputs matching the expected shapes and CUDA placement
    a = torch.rand(3, 2, dtype=torch.float).cuda()
    b = torch.rand(2, 3, dtype=torch.float).cuda()
    c = torch.rand(8000, dtype=torch.float).cuda()
    return (a, b, c)

