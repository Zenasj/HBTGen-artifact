# torch.rand(2, 2, dtype=torch.int64, device='cuda')  # Input shape and dtype for error reproduction
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = torch.as_tensor([0, 1, 1], dtype=torch.int64)
        self.cols = torch.as_tensor([0, 0, 0], dtype=torch.int64)
        self.values = torch.as_tensor([1, 1, 1], dtype=torch.int64)

    def forward(self, A):
        # Move indices and values to input device/dtype
        device = A.device
        dtype = A.dtype
        values = self.values.to(dtype=dtype, device=device)
        rows = self.rows.to(device=device)
        cols = self.cols.to(device=device)
        # Perform index_put_ operation that triggers the bug on CUDA + int64
        A.index_put_((rows, cols), values, accumulate=True)
        return A

def my_model_function():
    return MyModel()

def GetInput():
    # Returns CUDA tensor with int64 dtype to trigger the reported error
    return torch.zeros((2, 2), dtype=torch.int64, device='cuda')

