# torch.rand((), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Predefined reference containers to test membership behavior
        self.reference_list = [torch.tensor(1), 2]  # Contains tensor and integer
        self.reference_set = {torch.tensor(1), 2}   # Set with tensor and integer
        self.reference_tuple = (torch.tensor(1), 2) # Tuple with tensor and integer

    def forward(self, x):
        # Evaluate membership in all containers and return results as tensor
        in_list = x in self.reference_list
        in_set = x in self.reference_set
        in_tuple = x in self.reference_tuple
        return torch.tensor([in_list, in_set, in_tuple], dtype=torch.bool)

def my_model_function():
    # Returns model instance with predefined reference containers
    return MyModel()

def GetInput():
    # Returns 0D tensor matching the input expected by MyModel
    return torch.tensor(1, dtype=torch.int64)

