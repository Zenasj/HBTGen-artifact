# torch.rand(B, 8, dtype=torch.float32)  # Input shape inferred from the example's (128,8) input
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters matching the issue's example (weight: (10,4), bias: (1))
        self.weight = nn.Parameter(torch.rand(10, 4))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, input):
        # Compare CPU and GPU behaviors for linear layer
        cpu_ok = False
        try:
            # CPU computation (must fail if input.shape[1] != weight.shape[1])
            _ = torch.nn.functional.linear(input, self.weight, self.bias)
            cpu_ok = True
        except RuntimeError:
            pass

        gpu_ok = False
        try:
            # GPU computation (should fail only if dimensions match)
            _ = torch.nn.functional.linear(
                input.cuda(),
                self.weight.cuda(),
                self.bias.cuda()
            )
            gpu_ok = True
        except RuntimeError:
            pass

        # Return True if discrepancy exists (one succeeded, the other failed)
        return torch.tensor([cpu_ok != gpu_ok], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(128, 8, dtype=torch.float32)  # Matches input shape from issue example

