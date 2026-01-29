# torch.rand(1, 3, 28, 28, dtype=torch.float).to_sparse()  # Example input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        values = x.values()
        activated = F.relu(values)
        # Create a new sparse tensor with the same indices and size as input
        # Coalesce ensures gradients work correctly for sparse tensors
        return torch.sparse_coo_tensor(x.indices(), activated, x.size()).coalesce()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random sparse tensor input with shape (1, 3, 28, 28)
    dense = torch.rand(1, 3, 28, 28)
    sparse_input = dense.to_sparse()
    return sparse_input

