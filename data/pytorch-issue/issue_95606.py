# torch.rand(64, dtype=torch.int64)  # Input shape inferred from the issue's batch_size=64
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.randn(1))  # For device placement compatibility with torch.compile
    
    def forward(self, x):
        # Simulate worker processing: move to device and check for negative values
        device = self.dummy_param.device
        x = x.to(device)
        return x.lt(0).any()  # Return boolean indicating presence of negatives

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the original script's large_batch (64 elements)
    return torch.randint(100000, (64,), dtype=torch.int64)

