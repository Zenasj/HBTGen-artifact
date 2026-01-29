import math
import torch
from torch import nn

# torch.rand(B, 2, 2, dtype=torch.float32, device='cuda')  # Inferred input shape (batch, 2, 2)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuda_batch_size = (1 << 19) - 8  # CUDA's maximum allowed batch size for cholesky

    def forward(self, x):
        shape = x.shape
        batch_dims = shape[:-2]
        N = shape[-1]
        batch_size = math.prod(batch_dims)
        if batch_size <= self.cuda_batch_size:
            return torch.cholesky(x)
        else:
            x_flat = x.view(-1, N, N)
            acc = torch.empty_like(x_flat)
            for i in range(0, batch_size, self.cuda_batch_size):
                chunk_size = min(self.cuda_batch_size, batch_size - i)
                indices = torch.arange(i, i + chunk_size, device=x.device)
                chunk = x_flat[indices]
                acc[indices] = torch.cholesky(chunk)
            return acc.view(shape)

def my_model_function():
    return MyModel()

def GetInput():
    B = (1 << 19) - 7  # Exceeds CUDA's maximum batch size to trigger the workaround
    return torch.eye(2, device='cuda').expand(B, 2, 2).clone()  # Create contiguous PD tensor

