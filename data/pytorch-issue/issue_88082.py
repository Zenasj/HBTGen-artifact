# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        data_np = x.detach().cpu().numpy()
        rev = torch.utils.dlpack.from_dlpack(data_np)
        x.data = rev  # Triggers crash in PyTorch 1.13.0 with numpy 1.22-1.23
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32)

