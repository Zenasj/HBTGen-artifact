# torch.rand(B, 10, dtype=torch.float32)  # Assumed input shape based on test context and typical DDP model inputs
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simplified model structure based on DDP control flow test context
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        
    def forward(self, x):
        # Control flow condition that must be consistent across all ranks
        # The race condition causes divergent paths here
        if x.mean() > 0:
            return self.fc1(x)
        else:
            return self.fc2(x)

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected input shape
    return torch.rand(1, 10, dtype=torch.float32)  # B=1, 10 features (inferred from model's Linear layer)

