# torch.rand(1, dtype=torch.float32)  # Dummy input tensor (unused but required by structure)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create buffers with different dtypes sharing storage (empty tensors have same data_ptr=0)
        self.register_buffer('buffer_a', torch.tensor([], dtype=torch.int32))
        self.register_buffer('buffer_b', torch.tensor([], dtype=torch.int64))

    def forward(self, x):
        # Dummy forward to satisfy torch.compile requirements (input is not used)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Return dummy input tensor (shape/dtype doesn't affect the storage comparison issue)
    return torch.rand(1)

