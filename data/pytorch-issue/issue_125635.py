# torch.rand(5, dtype=torch.float32)
import torch
import copy
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Create a dictionary with Tensor views as values
        original_dict = {
            'k1': x[0],
            'k2': x[1],
            'k3': x[2],
            'k4': x[3],
            'k5': x[4]
        }
        # Trigger the IValue::deepcopy bug
        copied_dict = copy.deepcopy(original_dict)
        # Return the copied values as a tensor to observe the bug
        return torch.stack([copied_dict[key] for key in original_dict.keys()])

def my_model_function():
    return MyModel()

def GetInput():
    # Return a 1D tensor of shape (5,) with views that share storage
    return torch.rand(5, dtype=torch.float32)

