# torch.rand(5, 10)  # Inferred input shape (sum(idx_list)=5, features=10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, idx_list=[2, 3]):
        super(MyModel, self).__init__()
        self.idx_list = idx_list  # Split sizes must sum to input's first dimension

    def forward(self, x):
        splits = torch.split(x, self.idx_list, dim=0)  # Split along dimension 0
        # Example processing: return concatenated splits (to ensure output is a tensor)
        return torch.cat(splits, dim=0)  # Dummy operation to maintain output as tensor

def my_model_function():
    # Initialize with example idx_list [2,3] (sum=5)
    return MyModel(idx_list=[2, 3])

def GetInput():
    # Generate random input matching the required shape (sum(idx_list)=5, features=10)
    return torch.rand(5, 10)

