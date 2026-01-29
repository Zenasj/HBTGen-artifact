# torch.randint(0, 1000000, (B,), dtype=torch.int64)  # Inferred input shape from Dataset returning single index integers
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # Process 1D input (indices)

    def forward(self, x):
        # Convert index tensor to float and add channel dimension
        return self.fc(x.float().unsqueeze(1))

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random indices matching Dataset's output format
    batch_size = 4  # Example batch size
    return torch.randint(0, 1000000, (batch_size,), dtype=torch.int64)

