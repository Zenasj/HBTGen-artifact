# torch.rand(B, 10, dtype=torch.float32)  # Inferred input shape based on common FSDP use-cases (e.g., transformer or linear layers)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear model to replicate FSDP state_dict loading scenario
        self.layer = nn.Linear(10, 5)  # Example layer dimensions

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    # Returns a simple model instance
    return MyModel()

def GetInput():
    B = 4  # Example batch size
    return torch.rand(B, 10, dtype=torch.float32)

