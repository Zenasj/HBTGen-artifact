# torch.rand(B, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Replicating the model structure from the checkpoint's state_dict
        self.module = nn.Sequential(
            nn.Linear(32, 32),  # Matches "module.0.weight" in the ShardedTensor state
            nn.Identity(),      # Placeholder to align with "module.2" index in the state_dict
            nn.Linear(32, 8)    # Matches "module.2.weight" in the ShardedTensor state
        )
        
    def forward(self, x):
        return self.module(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generates input matching the model's expected dimensions (32 features)
    return torch.rand(2, 32, dtype=torch.float32)  # Batch size 2, 32 features

