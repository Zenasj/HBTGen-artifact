# torch.rand(64, ..., dtype=torch.float32, device='cuda:0')  # Input shape inferred from the bug's reproduction code
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare torch.argmax and torch.max outputs
        argmax_indices = torch.argmax(x, dim=0)
        max_indices = torch.max(x, dim=0)[1]
        # Return boolean indicating if all indices match
        return torch.all(argmax_indices == max_indices)

def my_model_function():
    return MyModel()

def GetInput():
    # Create input tensor exceeding 2^31 bytes to trigger the bug
    dtype = torch.float32  # Representative dtype from the issue's test
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = (2**31 // element_size) + 1  # Ensure size >2^31 bytes
    num_channel = 2 ** 6  # 64 channels as in the issue
    shape = (num_channel, numel // num_channel)
    t = torch.zeros(shape, dtype=dtype, device='cuda:0')
    t[num_channel // 2] = 1  # Set middle row to 1 for testing
    return t

