# torch.rand(B, 512, dtype=torch.float32)  # Inferred input shape based on example usage
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example layer to show forward flow, actual structure may vary
        self.linear = nn.Linear(512, 512)  # Arbitrary layer for demonstration

    def forward(self, x):
        x = self.cost_memory_function(x)
        # Simulate subsequent operations that may cause OOM due to retained memory
        # Actual operations may differ but this demonstrates the memory pressure scenario
        return self.linear(x)  # Example operation after memory-heavy function

    def cost_memory_function(self, x):
        """Memory-intensive function that creates large intermediate tensors"""
        # Creates a large tensor and returns a view that keeps it alive
        big_tensor = torch.rand(1000, 1000, device=x.device)
        # Return a slice (view) which keeps the entire big_tensor in memory
        return big_tensor[:x.shape[0], :x.shape[1]]  # Returns a view requiring full big_tensor retention

def my_model_function():
    # Returns an instance of the problematic model configuration
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected dimensions
    B = 1  # Batch size placeholder - can be adjusted
    return torch.rand(B, 512, dtype=torch.float32)

