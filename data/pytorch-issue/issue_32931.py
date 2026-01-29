# torch.rand(1, 2), torch.rand(1, 3)
import torch
from torch import nn

class OldStack(nn.Module):
    def forward(self, tensors):
        return torch.stack(tensors)

class NewStack(nn.Module):
    def forward(self, tensors):
        if not tensors:
            raise ValueError("Expected a non-empty list of tensors")
        first_size = tensors[0].size()
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.size() != first_size:
                raise RuntimeError(f"stack expects each tensor to be equal size, but got {first_size} at entry 0 and {tensor.size()} at entry {i}")
        return torch.stack(tensors)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.old_stack = OldStack()
        self.new_stack = NewStack()
    
    def forward(self, tensors):
        old_error = None
        new_error = None
        try:
            old_output = self.old_stack(tensors)
        except Exception as e:
            old_error = str(e)
        
        try:
            new_output = self.new_stack(tensors)
        except Exception as e:
            new_error = str(e)
        
        if old_error and new_error:
            return torch.tensor([old_error != new_error], dtype=torch.bool)
        elif old_error or new_error:
            return torch.tensor([True], dtype=torch.bool)
        else:
            return torch.tensor([torch.allclose(old_output, new_output)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1, 2), torch.rand(1, 3))

