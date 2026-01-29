# torch.rand(1, dtype=torch.float32)
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(X):
    return  # Empty kernel to trigger the error

class MyModel(torch.nn.Module):
    def forward(self, x):
        # Trigger the "No module named 'nvi'" error via num_ctas parameter
        kernel[(1,)](x, num_ctas=1)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

