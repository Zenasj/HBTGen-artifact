# torch.rand(10, 10, dtype=torch.float32)  # Input shape inferred from original test script
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy layer to trigger GPU computation
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        # Force computation on GPU to test device selection
        return self.linear(x.to('cuda'))

def my_model_function():
    # Initialize model with random weights
    model = MyModel()
    # Initialize weights for reproducibility (matching original test conditions)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    return model

def GetInput():
    # Generate input matching forward() expectations
    return torch.randn(10, 10, dtype=torch.float32)

