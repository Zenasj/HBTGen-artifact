# torch.rand(2, dtype=torch.int64)  # Inferred input shape: 1D tensor of length 2
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        target_device = x.device
        assert isinstance(target_device, torch.device)
        a = torch.zeros(2, 3, device=target_device)
        b = torch.zeros(2, 3, device=target_device)
        c = torch.zeros(2, 3, device=target_device)
        return a + b + c

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (2,), dtype=torch.int64, device=torch.device("cpu"))

