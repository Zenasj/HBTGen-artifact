# torch.rand(1)  # Dummy input tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Create sparse tensor as in the example
        indices = torch.zeros([1, 1], dtype=torch.long)
        values = torch.ones([1, 2, 3])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size([2, 2, 3]))
        
        # Modify original values tensor (in-place operation)
        values.resize_(4, 5)
        
        # Check if sparse tensor's values retain original size (new behavior)
        new_values_size = sparse_tensor.coalesce().values().size()
        expected_size = torch.Size([1, 2, 3])
        
        # Return boolean indicating if invariant holds (new behavior)
        return torch.tensor([int(new_values_size == expected_size)])

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input that matches the required input shape
    return torch.rand(1)

