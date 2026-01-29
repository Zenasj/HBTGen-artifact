# torch.rand(32, 64, 39, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(39, 6)  # Matches input_dim=39 and output_dim=6 from the issue

    def forward(self, x):
        # Compute full batch output
        full_output = self.linear(x)
        # Compute sliced input (first batch element) output
        sliced_x = x[0]  # Take first element in batch dimension
        sliced_output = self.linear(sliced_x)
        # Return difference between [0,0] of full output and [0] of sliced output (as in original print statements)
        return full_output[0, 0] - sliced_output[0]  # Directly outputs the discrepancy tensor

def my_model_function():
    model = MyModel()
    model.to("cuda")  # Replicate original CUDA setup
    return model

def GetInput():
    return torch.randn((32, 64, 39), dtype=torch.float32, device="cuda")  # Matches issue's tensor shape and device

