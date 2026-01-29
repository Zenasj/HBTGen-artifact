# torch.rand(0, 3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare CPU and CUDA outputs of max reduction along axis 1
        cpu_val, _ = torch.max(x.cpu(), 1)
        cuda_val, _ = torch.max(x.cuda(), 1)
        # Return 1 if shapes match (1 for success, 0 otherwise)
        return torch.tensor(
            cpu_val.shape == cuda_val.shape,
            dtype=torch.int32
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.empty(0, 3, 4, dtype=torch.float32)

