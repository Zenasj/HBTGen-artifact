# torch.rand(2, 2, dtype=torch.float32).to_sparse()  # Input shape: (2, 2) sparse COO tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # The model acts as a dummy to demonstrate sparse tensor handling
        # (e.g., in a distributed context like the PR's all_reduce example)
        self.identity = nn.Identity()  # Placeholder operation
    
    def forward(self, x):
        # Simple forward pass to comply with model structure requirements
        # Actual distributed operations like all_reduce are handled externally
        return self.identity(x)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generates a random sparse COO tensor matching the example's shape (2x2)
    dense = torch.rand(2, 2, dtype=torch.float32)
    sparse_tensor = dense.to_sparse()
    return sparse_tensor

