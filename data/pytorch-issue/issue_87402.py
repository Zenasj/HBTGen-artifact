import torch
from torch.distributions.constraints import lower_cholesky

# torch.rand(1, 1, dtype=torch.float32)  # Inferred input shape is (1,1)
class MyModel(torch.nn.Module):
    def forward(self, x):
        # Compute check on MPS (current device) and CPU, return comparison result
        mps_result = lower_cholesky.check(x)
        cpu_result = lower_cholesky.check(x.to("cpu"))
        return mps_result == cpu_result  # Returns tensor of shape () indicating match

def my_model_function():
    return MyModel()

def GetInput():
    # Generate valid 1x1 lower triangular matrix (any positive value suffices)
    return torch.rand(1, 1, dtype=torch.float32)

