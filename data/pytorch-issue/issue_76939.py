# torch.rand(1, 3, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Parameter(torch.randn(3).softmax(dim=-1))  # Shape (3,)

    def forward(self, x):
        log_probs = x.log_softmax(dim=-1)
        return F.cross_entropy(log_probs, self.target)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)  # 1D tensor to trigger no-batch-dimension error

