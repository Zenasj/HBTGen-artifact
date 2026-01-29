# torch.rand(3, 50, 50, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 5  # As specified in the issue's reproduction code

    def forward(self, x):
        # Compute on CPU
        x_cpu = x.cpu()  # Explicitly move to CPU
        values_cpu, indices_cpu = F.max_pool2d(
            x_cpu,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            return_indices=True,
        )
        
        # Compute on CUDA
        x_cuda = x.cuda()  # Move input to CUDA
        values_cuda, indices_cuda = F.max_pool2d(
            x_cuda,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            return_indices=True,
        )
        
        # Check if CUDA's indices shape matches values_cuda's shape (expected behavior)
        correct_cuda = torch.all(
            torch.tensor(indices_cuda.shape) == torch.tensor(values_cuda.shape)
        )
        # Return True if CUDA has a shape discrepancy (bug detected)
        return ~correct_cuda  # Returns a 0-dim boolean tensor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 50, 50, dtype=torch.float32)

