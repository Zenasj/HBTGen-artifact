# torch.rand(B, 768, dtype=torch.float32)  # Assuming input is a 1D embedding vector
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary preference output
        )
        # Placeholder for potential problematic submodule (if optimizer references external parameters)
        self.unmanaged_param = nn.Parameter(torch.zeros(768))  # Example of a parameter not part of FSDP hierarchy

    def forward(self, x):
        return self.policy_head(x)

def my_model_function():
    # FSDP wrapping would occur externally in user's code, but model structure remains
    model = MyModel()
    return model

def GetInput():
    # Random input matching model's expected embedding dimension
    return torch.rand(2, 768, dtype=torch.float32)  # Batch size 2, 768-dimensional embeddings

