# torch.rand(3, dtype=torch.float32)  # Output storage tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        input_tensor = torch.rand([3], dtype=torch.float32, device=x.device)
        return torch._C._special.special_scaled_modified_bessel_k1(out=x, x=input_tensor)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

