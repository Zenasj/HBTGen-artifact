# torch.rand(1, 2, dtype=torch.float32, device='cuda')  # Inferred input shape (B=1, C/H/W=2 as a 1D input)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        is_export = torch.onnx.is_in_onnx_export()
        factor = torch.tensor(is_export, dtype=x.dtype, device=x.device)
        return x * factor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, device='cuda', dtype=torch.float32)

