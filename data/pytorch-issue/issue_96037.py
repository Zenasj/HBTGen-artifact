# torch.rand(3, 2, 2, dtype=torch.float32)  # for each of the three input tensors (c, a, b)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        c, a, b = inputs
        # Create an uninitialized out tensor filled with NaNs
        out = torch.empty_like(c)
        out[:] = float('nan')
        # Perform baddbmm with beta=0 (should overwrite out entirely)
        result = torch.baddbmm(c, a, b, alpha=1, beta=0, out=out)
        # Return the number of NaNs remaining in the output (non-zero indicates bug)
        return result.isnan().sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate three tensors of shape (3,2,2) as required by the model
    c = torch.rand(3, 2, 2, dtype=torch.float32)
    a = torch.rand(3, 2, 2, dtype=torch.float32)
    b = torch.rand(3, 2, 2, dtype=torch.float32)
    return (c, a, b)

