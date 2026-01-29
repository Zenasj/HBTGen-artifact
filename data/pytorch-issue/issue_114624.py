# torch.rand(500, 500, dtype=torch.float32)  # Inferred input shape from test parameters N:500/M:500/P:500
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0  # Scaling factor for matrix multiplication result
        self.beta = 1.0   # Scaling factor for input tensor
        # Initialize matrices with sizes matching test parameters (N=500, M=500, P=500)
        self.mat1 = nn.Parameter(torch.randn(500, 500))  # (N, P)
        self.mat2 = nn.Parameter(torch.randn(500, 500))  # (P, M)

    def forward(self, input_tensor):
        # Perform addmm operation using the new implementation's workflow
        # Under the hood, this triggers the sequence: convert_channels_to_height_packed,
        # convert_channels_to_width_packed, mm, mul_scalar, add
        return torch.addmm(
            input=input_tensor,
            mat1=self.mat1,
            mat2=self.mat2,
            alpha=self.alpha,
            beta=self.beta
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor matching the shape expected by MyModel's forward()
    return torch.rand(500, 500, dtype=torch.float32)

