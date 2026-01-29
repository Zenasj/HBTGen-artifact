# torch.randint(45, 101, (1,), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, N_tensor):
        N = N_tensor.item()
        # Method 1: Using out= parameter (problematic approach)
        line_out = torch.zeros(1, N, dtype=torch.float32)
        torch.arange(-1, 1, 2/N, dtype=torch.float32, out=line_out)
        arr1 = line_out

        # Method 2: Without using out= (correct approach)
        arr2 = torch.arange(-1, 1, 2/N, dtype=torch.float32).view(1, -1)

        # Compare shape and values
        same_shape = (arr1.shape == arr2.shape)
        same_values = torch.allclose(arr1, arr2) if same_shape else False
        return torch.tensor([same_shape and same_values], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with N in [45, 100]
    return torch.randint(45, 101, (1,), dtype=torch.int32)

