# torch.rand(2000, 1, 2, dtype=torch.uint8)  # Input shape inferred from the repro example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        C = 0  # Third dimension index as per the repro setup
        non_contig_slice = x[:, 0, C]  # Create non-contiguous slice
        contig_slice = non_contig_slice.contiguous()  # Create contiguous version
        argmax_non = non_contig_slice.argmax()  # Argmax on non-contiguous slice
        argmax_cont = contig_slice.argmax()  # Argmax on contiguous slice
        # Return boolean indicating discrepancy between the two argmax results
        return torch.tensor([argmax_non != argmax_cont], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces the tensor setup from the issue's minimal example
    C = 0  # Third dimension index (fixed to 0 in this configuration)
    a = torch.zeros(2000, 1, 2, dtype=torch.uint8, device="cuda")  # Matches repro dimensions
    bc = torch.zeros(2000, dtype=torch.uint8, device="cuda")  # Base contiguous tensor
    bc[1355] = 1  # Set the correct argmax position
    a[:, 0, C] = bc  # Assign to create the non-contiguous slice
    return a

