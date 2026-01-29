# torch.rand(1, 512, 1024, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Original dense layer and pruned layer for comparison
        self.original = nn.Linear(1024, 1024)
        self.pruned = nn.Linear(1024, 1024)
        
        # Apply 30% sparsity to pruned model's weights (as per 0.3 sparsity in issue)
        mask = torch.rand(1024, 1024) > 0.3  # 70% non-zero
        self.pruned.weight.data *= mask  # Zero out 30% of weights

    def forward(self, x):
        # Run both models and return outputs for speed comparison
        orig_out = self.original(x)
        pruned_out = self.pruned(x)
        return orig_out, pruned_out  # Return tuple for performance measurement

def my_model_function():
    return MyModel()

def GetInput():
    # Random input matching LLM-like dimensions (batch, sequence length, hidden size)
    return torch.rand(1, 512, 1024, dtype=torch.float32)

