# torch.rand(B, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute logdet on CPU and CUDA, then compare outputs
        cpu_out = torch.logdet(x)
        cuda_x = x.cuda()
        cuda_out = torch.logdet(cuda_x)
        # Return element-wise comparison (True where outputs differ)
        return (cpu_out != cuda_out)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate batch of singular 2x2 matrices (determinant=0) to trigger discrepancy
    batch_size = 9  # Matches original test case
    base = torch.rand(batch_size, 1, 2)  # Random first row
    x = base.expand(batch_size, 2, 2)    # Duplicate as second row (determinant=0)
    return x

