# torch.rand(2, 4, dtype=torch.float32, device='cuda', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        x = torch.cat([x, x])  # Doubles the first dimension (2→4)
        out = torch.addmm(x, x, x).relu()  # addmm(input=x, mat1=x, mat2=x)
        # Return both the output tensor and the size (as a tensor for compatibility with compiled models)
        return out, torch.tensor([x.size(0)], dtype=torch.long, device=x.device)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 4, device="cuda", requires_grad=True)

